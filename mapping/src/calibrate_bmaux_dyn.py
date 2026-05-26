#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline calibration of the BMaux Std/Tdec from the QG1L model-error forcing.

For each consecutive reference timestep pair (η_ref^n, η_ref^{n+1}):
    η_pred = Qgm.step(h0=η_ref^n, hb=zeros, nstep=N_internal)
    ε^n = (η_ref^{n+1} − η_pred) / Δt_days   [units: m/day]

The residuals ε are projected onto the same anisotropic wavelet basis used by
BasisBMaux (via the shared helper _project_field_to_bmaux from calibrate_bmaux),
yielding Std and Tdec that describe the per-day forcing the QG model cannot
explain.

Output schema is identical to mapping/aux/aux_reduced_basis_BM.nc:
    coords: f (1/km), lon (deg), lat (deg)
    vars  : Std(f, lat, lon) [m/day], Tdec(f, lat, lon) [days]

Plug directly into BASIS_BMaux with factdec=1.0 (Std is already a forcing rate).

Usage:
    python -m mapping.src.calibrate_bmaux_dyn \\
        --ref '/data/.../eNATL60-BLB002_*SSH.nc' \\
        --out mapping/aux/aux_bmauxdyn_gulfstream.nc \\
        --name-ssh ssh \\
        --lon-min -80 --lon-max -30 --lat-min 32 --lat-max 44.5 \\
        --lmin 80 --lmax 1000 --facpsp 1.5 --npsp 3.5 --ntheta 4 \\
        --dlon-out 1 --dlat-out 1 --tdecmin 2.5 --tdecmax 40 \\
        --dt-internal 1200 --time-scheme rk3 --c0 2.7
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from importlib.machinery import SourceFileLoader
from typing import Optional, Sequence

import numpy as np
import xarray as xr

# Shared wavelet projection helpers (must match basis.py exactly)
from mapping.src.calibrate_bmaux import (
    _band_frequencies,
    _project_field_to_bmaux,
)

# Resolve paths relative to this file so the module is portable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # mapping/src
_MAPPING_DIR = os.path.dirname(_THIS_DIR)                        # mapping
_MASSH_ROOT = os.path.dirname(_MAPPING_DIR)                      # /home/flo/MASSH


# ----------------------------------------------------------------------------
# QG1L stepper helper  (Phase A)
# ----------------------------------------------------------------------------

def _build_qgm(
    lon_r: np.ndarray,
    lat_r: np.ndarray,
    dt_seconds: float,
    time_scheme: str,
    c_scalar: float,
    mdt: Optional[np.ndarray],
    ssh_template: Optional[np.ndarray],
):
    """Build a JIT-compiled Qgm instance from raw reference-grid arrays.

    Parameters
    ----------
    lon_r : (nx,) 1D longitude axis of the (cropped) reference domain (degrees).
    lat_r : (ny,) 1D latitude axis of the (cropped) reference domain (degrees).
    dt_seconds : float
        Internal model time step in seconds (default 1200 s; must satisfy CFL).
    time_scheme : str
        'Euler', 'rk2', or 'rk3'.
    c_scalar : float
        First-baroclinic phase speed in m/s.  Qgm takes nanmean internally,
        so passing a scalar is equivalent to a spatially uniform field.
    mdt : (ny, nx) float32 or None
        Mean Dynamic Topography on the reference grid.  If not None it is
        passed to Qgm so that step() internally adds MDT before PV inversion
        and subtracts it on exit.  Pass None to run in SLA-anomaly mode.
    ssh_template : (ny, nx) float32 or None
        First SSH frame used to build the land mask (NaN = land).

    Returns
    -------
    qgm : Qgm
        JIT-compiled model instance ready for step(h0, hb, nstep).
    """
    from mapping.src.grid import lonlat2dxdy

    # SourceFileLoader replicates how mod.py loads jqgm.py at runtime.
    # mapping/ must be in sys.path so jqgm.py's `from src.config import …` works.
    if _MAPPING_DIR not in sys.path:
        sys.path.insert(0, _MAPPING_DIR)
    dir_model = os.path.join(_MAPPING_DIR, "models", "model_qg1l")
    qgm_mod = SourceFileLoader("qgm", os.path.join(dir_model, "jqgm.py")).load_module()
    Qgm = qgm_mod.Qgm

    lon2d, lat2d = np.meshgrid(lon_r, lat_r)          # (ny, nx)
    dx, dy = lonlat2dxdy(lon2d, lat2d)                 # metres
    f_2d = 4.0 * np.pi / 86164.0 * np.sin(lat2d * np.pi / 180.0)

    qgm = Qgm(
        dx=dx,
        dy=dy,
        dt=float(dt_seconds),
        SSH=ssh_template,
        c=np.float32(c_scalar),
        f=f_2d,
        g=9.81,
        mdt=mdt,
        bathymetry_PV_term=None,
        time_scheme=time_scheme,
        compile=True,
    )
    return qgm


# ----------------------------------------------------------------------------
# Residual time series  (Phase B)
# ----------------------------------------------------------------------------

def _to_days_since_epoch(time_da) -> np.ndarray:
    """Convert an xarray time DataArray to float64 days since 1970-01-01.

    Handles datetime64, cftime objects, and raw CF-encoded numerics.
    """
    vals = time_da.values
    if np.issubdtype(np.asarray(vals).dtype, np.datetime64):
        _epoch = np.datetime64("1970-01-01", "ns")
        return (vals.astype("datetime64[ns]") - _epoch).astype(np.float64) / 86400e9
    # cftime or plain objects — try via pandas Timestamp (handles most calendars)
    try:
        import pandas as pd
        return np.array(
            [(pd.Timestamp(str(v)) - pd.Timestamp("1970-01-01")).total_seconds() / 86400.0
             for v in vals],
            dtype=np.float64,
        )
    except Exception:
        pass
    # Last resort: decode raw CF numerics using cftime
    import cftime
    units = time_da.attrs.get("units", "seconds since 1970-01-01")
    calendar = time_da.attrs.get("calendar", "standard")
    decoded = cftime.num2date(vals, units=units, calendar=calendar)
    _epoch_ord = cftime.datetime(1970, 1, 1, calendar=calendar).toordinal()
    return np.array([d.toordinal() - _epoch_ord for d in decoded], dtype=np.float64)


def compute_qg_residuals(
    ssh: np.ndarray,
    t_days: np.ndarray,
    qgm,
    hb_for_step: Optional[np.ndarray],
    dt_internal_sec: float,
    forecast_steps: int = 1,
    h0_hb_series: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> tuple:
    """Compute QG1L model-error forcing residuals from a reference SSH stack.

    For each pair (ssh^n, ssh^{n+k}) where k = forecast_steps:
        if h0_hb_series is None:
            η_pred = qgm.step(h0=ssh^n, hb=hb_for_step, nstep=N*k)
        else:
            η_pred = qgm.step(h0=h0_hb_series^n, hb=h0_hb_series^n, nstep=N*k)
        ε^n = (ssh^{n+k} − η_pred) / (k·Δt_days)   [units: m/day]

    Longer forecast horizons let the QG model error accumulate, producing
    larger residuals that better reflect the amplitude the basis must correct.

    Parameters
    ----------
    ssh : (nt, ny, nx) float64
        Reference SSH (truth).  NaN on land; used as-is (no pre-zeroing).
    t_days : (nt,) float
        Time axis in days.
    qgm : Qgm
        Pre-built QG model instance (from _build_qgm).
    hb_for_step : (ny, nx) float32 or None
        Static background SSH for hb.  Used only when h0_hb_series is None.
        None → zeros (SLA mode).
    dt_internal_sec : float
        Internal model time step (seconds); nstep = round(Δt_ref / dt_internal).
    forecast_steps : int
        Number of reference time steps to forecast ahead (default 1).  Larger
        values let QG errors accumulate, yielding larger and more physically
        meaningful residuals.  Output has shape (nt - forecast_steps, ny, nx).
    h0_hb_series : (nt, ny, nx) float32 or None
        Time-varying prescribed SSH.  When provided, each QG forward is
        initialised with ``h0_hb_series[n]`` and the same field is used as
        the boundary condition (hb), replacing both ``ssh[n]`` (h0) and
        ``hb_for_step`` (hb).  The residual is always measured against the
        reference ``ssh`` (truth).  None = legacy behaviour.
    verbose : bool

    Returns
    -------
    eps : (nt-forecast_steps, ny, nx) float64
        Forcing residuals in m/day.  NaN on land.
    t_eps : (nt-forecast_steps,) float
        Time axis for residuals (same as t_days[:nt-forecast_steps]).
    """

    nt, ny, nx = ssh.shape
    k = max(1, int(forecast_steps))
    if nt <= k:
        raise ValueError(f"forecast_steps={k} >= nt={nt}; need more time steps.")
    dt_ref_days = float(np.median(np.diff(t_days)))
    dt_ref_sec = dt_ref_days * 86400.0
    nstep_per_ref = max(1, round(dt_ref_sec / dt_internal_sec))
    nstep_total = nstep_per_ref * k
    if verbose:
        print(
            f"[calibrate_bmaux_dyn] computing QG residuals: "
            f"nt={nt}, dt_ref={dt_ref_days:.2f}d, "
            f"forecast={k} steps ({k*dt_ref_days:.1f}d), "
            f"nstep_total={nstep_total} × {dt_internal_sec:.0f}s"
        )

    hb_static = (
        hb_for_step
        if hb_for_step is not None
        else np.zeros((ny, nx), dtype=np.float32)
    )

    land_mask = ~np.isfinite(ssh[0])
    eps = np.full((nt - k, ny, nx), np.nan, dtype=np.float64)
    forecast_days_total = k * dt_ref_days

    for n in range(nt - k):
        if verbose:
            print("[calibrate_bmaux_dyn]   step {}/{} (t={:.1f}d)".format(n + 1, nt - k, t_days[n]))
        if h0_hb_series is not None:
            h0_raw = h0_hb_series[n].astype(np.float32)
            if not np.any(np.isfinite(h0_raw)):
                # Insufficient hb coverage — fall back to reference SSH for this step
                h0 = ssh[n].astype(np.float32)
                h0_clean = np.where(np.isfinite(h0), h0, 0.0).astype(np.float32)
                hb = hb_static
            else:
                h0_clean = np.where(np.isfinite(h0_raw), h0_raw, 0.0).astype(np.float32)
                hb = h0_clean  # prescribed field used as both initial state and boundary
        else:
            h0 = ssh[n].astype(np.float32)
            h0_clean = np.where(np.isfinite(h0), h0, 0.0).astype(np.float32)
            hb = hb_static
        if verbose and n == 0:
            print(
                f"[calibrate_bmaux_dyn]     first JIT call (nstep={nstep_total}): "
                "XLA compilation may take several minutes — not stuck."
            )
        eta_pred = np.array(qgm.step_jit(h0_clean, hb, nstep_total))
        residual = (ssh[n + k] - eta_pred.astype(np.float64)) / forecast_days_total
        residual[land_mask] = np.nan
        eps[n] = residual
        if verbose and n == 0:
            # First residual: check magnitude as sanity gate.
            rms = float(np.nanstd(residual))
            print(f"[calibrate_bmaux_dyn]   first residual RMS = {rms:.4f} m/day")

    t_eps = t_days[:nt - k].copy()
    return eps, t_eps


# ----------------------------------------------------------------------------
# Main calibration function  (Phase D)
# ----------------------------------------------------------------------------

def calibrate_bmaux_dyn(
    ref_path,
    out_path: str,
    name_var: Optional[dict] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lmin: float = 80.0,
    lmax: float = 1000.0,
    facpsp: float = 1.5,
    npsp: float = 3.5,
    ntheta: int = 4,
    dlon_out: float = 1.0,
    dlat_out: float = 1.0,
    tdecmin: float = 2.5,
    tdecmax: float = 40.0,
    valid_min: float = 0.5,
    dt_internal: float = 1200.0,
    time_scheme: str = "rk3",
    c0: float = 2.7,
    filec: Optional[str] = None,
    name_var_c: Optional[dict] = None,
    path_mdt: Optional[str] = None,
    name_var_mdt: Optional[dict] = None,
    lon_unit_out: str = "0_360",
    forecast_days: Optional[float] = None,
    out_eps_path: Optional[str] = None,
    in_eps_path: Optional[str] = None,
    path_hb: Optional[str] = None,
    name_var_hb: Optional[dict] = None,
    min_hb_valid_frac: float = 0.0,
    regrid_qg: bool = False,
    n_jobs: int = 1,
    verbose: bool = True,
    tdec_bins=None,
) -> xr.Dataset:
    """Calibrate BMaux Std/Tdec from the QG1L model-error forcing residuals.

    All parameters shared with calibrate_bmaux() behave identically.
    Extra QG parameters:

    dt_internal : float
        Internal QG time step in seconds (default 1200 s; must satisfy CFL).
    time_scheme : str
        'Euler', 'rk2', or 'rk3'.
    c0 : float
        First-baroclinic phase speed in m/s (used if filec is None).
    filec : str or None
        Path to NetCDF with 2D phase-speed field.
    name_var_c : dict or None
        Variable names in filec: {'lon':…, 'lat':…, 'var':…}.
        Defaults to {'lon':'lon', 'lat':'lat', 'var':'c1'} (ships with MASSH).
    path_mdt : str or None
        Path to MDT NetCDF.  When provided, Qgm is built with MDT so the
        model operates on full SSH (MDT+SLA).
    name_var_mdt : dict or None
        Variable names in path_mdt: {'lon':…, 'lat':…, 'var':…}.
        Defaults to {'lon':'longitude', 'lat':'latitude', 'var':'mdt'}
        (matches aux_mdt_cnes_cls18_global.nc shipped with MASSH).
    forecast_days : float or None
        Forecast horizon in days used to compute residuals.  None (default)
        means one reference time step.  Longer horizons let QG errors
        accumulate, producing larger Std values that need less facQ correction.
    out_eps_path : str or None
        If provided, save the raw QG residuals as a NetCDF file at this path
        before the wavelet projection step.  Useful for diagnosing calibration.
    in_eps_path : str or None
        If provided and the file exists, load pre-computed QG residuals from
        this NetCDF file (written by a previous run with ``out_eps_path``) and
        skip the reference-SSH loading, QGM construction, and residual
        computation entirely.  Calibration metadata (c0, dt_internal, …) is
        recovered from the file attributes.  ``ref_path`` is ignored.
    path_hb : str or None
        Path to a NetCDF file containing a time-varying SSH field with shape
        ``(time, lat, lon)``, typically the analysis or background SSH from a
        previous experiment.  For each QG forward step at time ``t_n``, the
        nearest time slice is interpolated to the reference grid and used as
        **both** the initial condition (``h0``) and the boundary condition
        (``hb``).  The residual is still measured against the reference SSH
        (truth).  When None, ``h0 = ssh[n]`` (reference) and ``hb = zeros``.
        The time axis in ``path_hb`` must be in the same numeric units (e.g.
        days since the same epoch) as the reference SSH.
    name_var_hb : dict or None
        Variable names in ``path_hb``:
        {{'time':…, 'lon':…, 'lat':…, 'var':…}}.
        Defaults to {{'time':'time', 'lon':'lon', 'lat':'lat', 'var':'ssh'}}.
    min_hb_valid_frac : float
        Minimum fraction of ocean points (where ``valid_mask`` is True) that
        must be non-NaN in a hb time slice for it to be used.  Reference times
        whose interpolated hb coverage falls below this threshold are flagged
        as bad: ``h0_hb_series[n]`` is set to all-NaN and ``compute_qg_residuals``
        falls back to ``ssh[n]`` / zeros for that step (default 0.0 = no filter).
    regrid_qg : bool
        If True, the reference SSH is interpolated to a uniform longitude grid
        with ``dlon = dlat / cos(lat_mean)`` before running the QG model, so
        that ``dx ≈ dy`` in metres at the domain centre.  Residuals are then
        projected back to the original reference grid.  Recommended when the
        domain spans several degrees of latitude and varying ``dx`` would
        otherwise distort the QG inversion (default False).
    n_jobs : int
        Number of parallel worker processes for the wavelet projection
        (default 1 = serial).  Set to -1 to use all available CPUs.
    """
    if name_var is None:
        name_var = {"time": "time", "lon": "lon", "lat": "lat", "ssh": "ssh"}
    if name_var_c is None:
        name_var_c = {"lon": "lon", "lat": "lat", "var": "c1"}
    if name_var_mdt is None:
        name_var_mdt = {"lon": "longitude", "lat": "latitude", "var": "mdt"}
    if name_var_hb is None:
        name_var_hb = {"time": "time", "lon": "lon", "lat": "lat", "var": "ssh"}

    # ---- Load pre-computed residuals OR run full QGM pipeline -------------
    if in_eps_path is not None:
        # Skip reference-SSH loading, QGM construction, and residual computation.
        if not os.path.exists(in_eps_path):
            raise FileNotFoundError(f"--in-eps file not found: {in_eps_path}")
        if verbose:
            print(f"[calibrate_bmaux_dyn] loading pre-computed residuals from {in_eps_path}")
        _ds_in  = xr.open_dataset(in_eps_path, decode_times=True)
        eps     = _ds_in["eps"].values.astype(np.float64)
        t_eps   = _to_days_since_epoch(_ds_in["time"])
        lon_r   = _ds_in["lon"].values.astype(np.float64)
        lat_r   = _ds_in["lat"].values.astype(np.float64)
        valid_mask     = np.any(np.isfinite(eps), axis=0)
        eps_clean      = np.where(np.isfinite(eps), eps, 0.0)
        c_val          = float(_ds_in.attrs.get("c0_ms",         c0))
        dt_internal    = float(_ds_in.attrs.get("dt_internal_s", dt_internal))
        time_scheme    = str(  _ds_in.attrs.get("time_scheme",   time_scheme))
        forecast_steps = int(  _ds_in.attrs.get("forecast_steps", 1))
        _mdt_used = _ds_in.attrs.get("mdt_used", "none")
        path_mdt = None if _mdt_used in ("none", "", None) else str(_mdt_used)
        files = [in_eps_path]
        if lon_min is None: lon_min = float(lon_r.min())
        if lon_max is None: lon_max = float(lon_r.max())
        if lat_min is None: lat_min = float(lat_r.min())
        if lat_max is None: lat_max = float(lat_r.max())
        if verbose:
            _rms = float(np.nanstd(eps[np.isfinite(eps)]))
            print(f"[calibrate_bmaux_dyn] eps {eps.shape}, RMS = {_rms:.4f} m/day")
    else:
        nm_t = name_var["time"]
        nm_lon = name_var["lon"]
        nm_lat = name_var["lat"]
        nm_ssh = name_var["ssh"]

        # ---- Load reference SSH --------------------------------------------
        if ref_path is None:
            raise ValueError("ref_path is required when in_eps_path is not provided")
        if isinstance(ref_path, (list, tuple)):
            files = list(ref_path)
        else:
            files = sorted(glob.glob(ref_path)) if any(c in str(ref_path) for c in "*?[") else [ref_path]
        if not files:
            raise FileNotFoundError(f"No reference files match: {ref_path}")
        if verbose:
            print(f"[calibrate_bmaux_dyn] opening {len(files)} reference file(s)")
        if len(files) > 1:
            ds = xr.open_mfdataset(files, combine="by_coords")
        else:
            ds = xr.open_dataset(files[0])

        lon = ds[nm_lon].values
        lat = ds[nm_lat].values
        if lon.ndim != 1 or lat.ndim != 1:
            raise ValueError("Reference lon/lat must be 1D (regular grid).")

        # Time → days since first record
        t_da = ds[nm_t]
        t_vals = t_da.values
        if np.issubdtype(t_vals.dtype, np.datetime64):
            t_sec = (t_vals - t_vals[0]) / np.timedelta64(1, "s")
        else:
            units = t_da.attrs.get("units", "seconds since 1970-01-01")
            if "day" in units.lower():
                t_sec = (t_vals - t_vals[0]) * 86400.0
            elif "hour" in units.lower():
                t_sec = (t_vals - t_vals[0]) * 3600.0
            else:
                t_sec = t_vals - t_vals[0]
        t_days = t_sec / 86400.0
        nt = t_days.size
        if nt < 4:
            raise ValueError(f"Need at least 4 time steps, got {nt}")
        dt_days = float(np.median(np.diff(t_days)))
        # Absolute time axis (days since 1970-01-01) — used for hb intersection and output dates
        t_ref_abs = _to_days_since_epoch(t_da)
        if verbose:
            print(f"[calibrate_bmaux_dyn] {nt} time steps, dt = {dt_days:.3f} days")

        # ---- Region crop ---------------------------------------------------
        ref_lon_unit = "0_360" if np.nanmin(lon) >= 0 and np.nanmax(lon) > 180 else "-180_180"
        if lon_min is None:
            lon_min = float(np.nanmin(lon))
        if lon_max is None:
            lon_max = float(np.nanmax(lon))
        if lat_min is None:
            lat_min = float(np.nanmin(lat))
        if lat_max is None:
            lat_max = float(np.nanmax(lat))

        if ref_lon_unit == "0_360":
            lon_min_ref = lon_min % 360
            lon_max_ref = lon_max % 360
            if lon_max_ref < lon_min_ref:
                lon_max_ref += 360
        else:
            lon_min_ref = ((lon_min + 180) % 360) - 180
            lon_max_ref = ((lon_max + 180) % 360) - 180

        j_sel = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        i_sel = np.where((lon >= lon_min_ref) & (lon <= lon_max_ref))[0]
        if i_sel.size < 4 or j_sel.size < 4:
            raise ValueError("Region crop too small (need >=4 pixels each direction).")
        lon_r = lon[i_sel]
        lat_r = lat[j_sel]

        ssh = ds[nm_ssh].isel({nm_lat: j_sel, nm_lon: i_sel}).values.astype(np.float64)
        if ssh.ndim != 3:
            raise ValueError(f"SSH must be 3D (time,lat,lon), got shape {ssh.shape}")
        if ssh.shape != (nt, lat_r.size, lon_r.size):
            dim_order = ds[nm_ssh].dims
            order = [dim_order.index(d) for d in (nm_t, nm_lat, nm_lon)]
            ssh = np.transpose(ssh, order)
        valid_mask = np.isfinite(ssh[0])
        if verbose:
            print(f"[calibrate_bmaux_dyn] SSH cropped to ({nt}, {lat_r.size}, {lon_r.size})")

        # ---- Phase speed ---------------------------------------------------
        if filec is not None and os.path.exists(filec):
            from mapping.src.grid import interp2d
            ds_c = xr.open_dataset(filec)
            lon2d, lat2d = np.meshgrid(lon_r, lat_r)
            c_2d = interp2d(ds_c, name_var_c, lon2d, lat2d)
            c_val = float(np.nanmean(c_2d))
            if verbose:
                print(f"[calibrate_bmaux_dyn] c loaded from {filec}: mean = {c_val:.3f} m/s")
        else:
            c_val = float(c0)
            if verbose:
                print(f"[calibrate_bmaux_dyn] using uniform c0 = {c_val} m/s")

        # ---- MDT -----------------------------------------------------------
        mdt_r = None
        if path_mdt is not None and os.path.exists(path_mdt):
            from mapping.src.grid import interp2d
            ds_mdt = xr.open_dataset(path_mdt)
            lon2d, lat2d = np.meshgrid(lon_r, lat_r)
            mdt_r = interp2d(ds_mdt, name_var_mdt, lon2d, lat2d).astype(np.float32)
            mdt_r = np.where(np.isfinite(mdt_r), mdt_r, 0.0).astype(np.float32)
            if verbose:
                print(f"[calibrate_bmaux_dyn] MDT loaded from {path_mdt}")

        # ---- Prescribed h0/hb series (time-varying, from previous experiment) --
        # Shape after loading: (nt_ref, ny_r, nx_r); used as both h0 and hb.
        h0_hb_series = None
        hb_for_step = None  # unused when h0_hb_series is not None
        if path_hb is not None:
            from mapping.src.grid import interp2d
            from scipy.interpolate import interp1d as _interp1d
            time_name_hb = name_var_hb.get("time", "time")
            _name_var_2d_hb = {k: v for k, v in name_var_hb.items() if k != "time"}

            # Open with time decoding so calendars / units are handled correctly
            if isinstance(path_hb, (list, tuple)):
                _hb_files = list(path_hb)
            else:
                _hb_files = sorted(glob.glob(path_hb)) if any(c in str(path_hb) for c in "*?[") else [path_hb]
            if not _hb_files:
                raise FileNotFoundError(f"--path-hb pattern matched no files: {path_hb}")
            if len(_hb_files) > 1:
                ds_hb = xr.open_mfdataset(_hb_files, combine="by_coords", decode_times=True)
            else:
                ds_hb = xr.open_dataset(_hb_files[0], decode_times=True)

            # Decode hb time axis to float days since 1970-01-01
            t_hb_abs = _to_days_since_epoch(ds_hb[time_name_hb])  # (nt_hb,)
            nt_hb = len(t_hb_abs)

            # Clip ref time axis to the intersection with the hb time window
            _t_int_start = max(float(t_ref_abs[0]), float(t_hb_abs[0]))
            _t_int_end   = min(float(t_ref_abs[-1]), float(t_hb_abs[-1]))
            if _t_int_start >= _t_int_end:
                raise ValueError(
                    f"No temporal overlap between ref "
                    f"[{t_ref_abs[0]:.3f}, {t_ref_abs[-1]:.3f}] and hb "
                    f"[{t_hb_abs[0]:.3f}, {t_hb_abs[-1]:.3f}] (days since 1970-01-01)."
                )
            _ref_sel = (t_ref_abs >= _t_int_start) & (t_ref_abs <= _t_int_end)
            if not _ref_sel.all():
                _n_head = int((t_ref_abs < _t_int_start).sum())
                _n_tail = int((t_ref_abs > _t_int_end).sum())
                ssh        = ssh[_ref_sel]
                t_days     = t_days[_ref_sel]
                t_ref_abs  = t_ref_abs[_ref_sel]
                nt         = len(t_days)
                valid_mask = np.isfinite(ssh[0])
                if verbose:
                    import pandas as _pd
                    _d0 = (_pd.Timestamp("1970-01-01") + _pd.Timedelta(days=_t_int_start)).date()
                    _d1 = (_pd.Timestamp("1970-01-01") + _pd.Timedelta(days=_t_int_end)).date()
                    print(
                        f"[calibrate_bmaux_dyn] hb: clipped ref to intersection "
                        f"[{_d0}, {_d1}] "
                        f"(removed {_n_head} step(s) at start, {_n_tail} at end) "
                        f"→ nt_ref={nt}"
                    )

            # Spatial interpolation: loop over all hb time steps
            lon2d_hb, lat2d_hb = np.meshgrid(lon_r, lat_r)
            hb_on_refgrid = np.zeros((nt_hb, len(lat_r), len(lon_r)), dtype=np.float32)
            _n_ocean = int(valid_mask.sum())
            hb_valid_frac = np.zeros(nt_hb, dtype=np.float64)
            for _it in range(nt_hb):
                _ds_sl = ds_hb.isel({time_name_hb: _it})
                _sl = interp2d(_ds_sl, _name_var_2d_hb, lon2d_hb, lat2d_hb).astype(np.float32)
                hb_valid_frac[_it] = float(np.isfinite(_sl)[valid_mask].mean()) if _n_ocean > 0 else 1.0
                hb_on_refgrid[_it] = np.where(np.isfinite(_sl), _sl, 0.0)

            # Temporal interpolation: linear, clamped at boundaries
            _f_time = _interp1d(
                t_hb_abs, hb_on_refgrid, axis=0, kind="linear",
                bounds_error=False,
                fill_value=(hb_on_refgrid[0], hb_on_refgrid[-1]),
            )
            h0_hb_series = _f_time(t_ref_abs).astype(np.float32)

            # Temporal interpolation of valid fraction
            _f_frac = _interp1d(
                t_hb_abs, hb_valid_frac, kind="linear",
                bounds_error=False, fill_value=(hb_valid_frac[0], hb_valid_frac[-1]),
            )
            hb_valid_frac_ref = _f_frac(t_ref_abs)

            # Flag reference times with insufficient ocean coverage → all-NaN sentinel
            if min_hb_valid_frac > 0.0:
                _bad = hb_valid_frac_ref < min_hb_valid_frac
                if _bad.any():
                    h0_hb_series[_bad] = np.nan
                    if verbose:
                        print(
                            f"[calibrate_bmaux_dyn]   {int(_bad.sum())} reference time steps "
                            f"have hb ocean coverage < {min_hb_valid_frac:.2f} "
                            "→ falling back to ssh[n] for those steps"
                        )

            if verbose:
                _max_gap = float(np.max(np.abs(
                    t_ref_abs - t_hb_abs[
                        np.argmin(np.abs(t_hb_abs[:, None] - t_ref_abs[None, :]), axis=0)
                    ]
                )))
                print(
                    f"[calibrate_bmaux_dyn] prescribed h0/hb loaded from {path_hb}: "
                    f"nt_hb={nt_hb}, nt_ref={len(t_days)}, "
                    f"max_nearest_gap={_max_gap:.3f} days, "
                    f"mean_coverage={float(hb_valid_frac_ref.mean()):.2f}, "
                    f"mean={float(np.nanmean(h0_hb_series)):.4f} m, "
                    f"std={float(np.nanstd(h0_hb_series)):.4f} m"
                )


        # ---- Optionally regrid SSH to uniform dx/dy for QG model -----------
        # Only the longitude axis is adjusted (dlon = dlat/cos(lat_mean)) so that
        # dx ≈ dy in metres at the domain centre.  lat_r is kept unchanged.
        lon_r_orig      = lon_r.copy()
        lat_r_orig      = lat_r.copy()
        valid_mask_orig = valid_mask.copy()

        if regrid_qg:
            from scipy.interpolate import interp1d as _interp1d_lon
            _dlat     = float(np.median(np.diff(lat_r)))
            _dlon_qg  = _dlat / float(np.cos(np.deg2rad(float(np.mean(lat_r)))))
            _n_lon_qg = max(4, round((float(lon_r[-1]) - float(lon_r[0])) / _dlon_qg) + 1)
            lon_qg    = np.linspace(float(lon_r[0]), float(lon_r[-1]), _n_lon_qg)
            if verbose:
                print(
                    f"[calibrate_bmaux_dyn] regrid_qg: nx {lon_r.size} → {_n_lon_qg} "
                    f"(dlon={_dlon_qg:.4f}°), dx≈dy≈{111.0 * _dlat:.1f} km "
                    f"at lat_mean={np.mean(lat_r):.1f}°"
                )

            def _interp_to_qg(field_2d):
                """Interpolate (ny, nx_orig) → (ny, nx_qg) row-by-row in longitude."""
                _clean = np.where(np.isfinite(field_2d), field_2d, 0.0)
                return np.stack([
                    _interp1d_lon(lon_r_orig, _clean[_j], kind='linear',
                                  bounds_error=False, fill_value=0.0)(lon_qg)
                    for _j in range(len(lat_r_orig))
                ])

            def _interp_from_qg(field_2d):
                """Interpolate (ny, nx_qg) → (ny, nx_orig) row-by-row in longitude."""
                _clean = np.where(np.isfinite(field_2d), field_2d, 0.0)
                return np.stack([
                    _interp1d_lon(lon_qg, _clean[_j], kind='linear',
                                  bounds_error=False, fill_value=0.0)(lon_r_orig)
                    for _j in range(len(lat_r_orig))
                ])

            valid_mask_qg = _interp_to_qg(valid_mask.astype(np.float64)) > 0.5
            ssh_qg = np.stack([_interp_to_qg(ssh[n]) for n in range(nt)], axis=0)
            ssh_qg[:, ~valid_mask_qg] = np.nan

            if h0_hb_series is not None:
                _h0_hb_qg = np.full((nt, len(lat_r), _n_lon_qg), np.nan, dtype=np.float32)
                for _n in range(nt):
                    if np.any(np.isfinite(h0_hb_series[_n])):
                        _h0_hb_qg[_n] = _interp_to_qg(h0_hb_series[_n]).astype(np.float32)
                    # else: preserve all-NaN sentinel (min_hb_valid_frac fallback)
                h0_hb_series = _h0_hb_qg

            lon_r      = lon_qg
            ssh        = ssh_qg
            valid_mask = valid_mask_qg

        # ---- Build QG model ------------------------------------------------
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # must be set before JAX init
        ssh_template = np.where(valid_mask, ssh[0].astype(np.float32), np.nan)
        if verbose:
            print(
                f"[calibrate_bmaux_dyn] building Qgm "
                f"(c={c_val:.2f} m/s, dt_internal={dt_internal:.0f}s, "
                f"scheme={time_scheme}, mdt={'yes' if mdt_r is not None else 'no'}, "
                f"hb={'prescribed' if h0_hb_series is not None else 'zeros'}")
        qgm = _build_qgm(
            lon_r, lat_r,
            dt_seconds=dt_internal,
            time_scheme=time_scheme,
            c_scalar=c_val,
            mdt=mdt_r,
            ssh_template=ssh_template,
        )

        # ---- Compute QG residuals ------------------------------------------
        dt_ref_days_approx = float(np.median(np.diff(t_days)))
        if forecast_days is not None and forecast_days > 0:
            forecast_steps = max(1, round(forecast_days / dt_ref_days_approx))
        else:
            forecast_steps = 1
        if verbose and forecast_steps > 1:
            print(
                f"[calibrate_bmaux_dyn] forecast horizon = {forecast_steps} steps "
                f"({forecast_steps * dt_ref_days_approx:.1f} d)"
            )
        eps, t_eps = compute_qg_residuals(
            ssh, t_days, qgm, hb_for_step, dt_internal,
            forecast_steps=forecast_steps,
            h0_hb_series=h0_hb_series,
            verbose=verbose,
        )

        # Back-project residuals from QG grid to original reference grid
        if regrid_qg:
            _eps_orig = np.stack(
                [_interp_from_qg(eps[_n]) for _n in range(eps.shape[0])], axis=0
            )
            _eps_orig[:, ~valid_mask_orig] = np.nan
            eps        = _eps_orig
            lon_r      = lon_r_orig
            valid_mask = valid_mask_orig
            if verbose:
                print(
                    f"[calibrate_bmaux_dyn] regrid_qg: residuals projected back to "
                    f"reference grid (lon: {len(lon_qg)} → {lon_r.size})"
                )

        # NaN→0 for projection; valid_mask tracks ocean
        eps_clean = np.where(np.isfinite(eps), eps, 0.0)

        # ---- Optionally save raw residuals ---------------------------------
        if out_eps_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(out_eps_path)) or ".", exist_ok=True)
            _t_eps_abs   = t_ref_abs[:len(t_eps)]
            _t_eps_dates = (
                np.datetime64("1970-01-01", "ns")
                + (_t_eps_abs * 86400.0 * 1e9).astype("int64").view("timedelta64[ns]")
            )
            ds_eps = xr.Dataset(
                data_vars={"eps": (("time", "lat", "lon"), eps.astype(np.float32))},
                coords={
                    "time": ("time", _t_eps_dates),
                    "lat": ("lat", lat_r.astype(np.float32)),
                    "lon": ("lon", lon_r.astype(np.float32)),
                },
                attrs={
                    "description": "QG1L model-error forcing residuals eps = (ssh[n+k] - QG_pred) / (k*dt)",
                    "units": "m/day",
                    "c0_ms": c_val,
                    "dt_internal_s": dt_internal,
                    "time_scheme": time_scheme,
                    "forecast_steps": forecast_steps,
                    "mdt_used": path_mdt if path_mdt else "none",
                    "hb_used": path_hb if path_hb else "none",
                    "source_files": ";".join(files),
                },
            )
            ds_eps["eps"].attrs["units"] = "m/day"
            ds_eps["eps"].attrs["long_name"] = "QG forcing residual"
            ds_eps["lat"].attrs["units"] = "degrees_north"
            ds_eps["lon"].attrs["units"] = "degrees_east"
            ds_eps.to_netcdf(out_eps_path)
            if verbose:
                print(f"[calibrate_bmaux_dyn] residuals saved to {out_eps_path}")

        if verbose:
            eps_ocean = eps[np.isfinite(eps)]
            if eps_ocean.size > 0:
                rms = float(np.std(eps_ocean))
                print(f"[calibrate_bmaux_dyn] global residual RMS = {rms:.4f} m/day")
                if rms > 0.5:
                    print(
                        "[calibrate_bmaux_dyn] WARNING: residual RMS > 0.5 m/day — "
                        "check CFL (try smaller --dt-internal) or model parameters."
                    )

    # ---- Output coarse grid ------------------------------------------------
    lon_out = np.arange(lon_min, lon_max + 0.5 * dlon_out, dlon_out, dtype=np.float64)
    lat_out = np.arange(lat_min, lat_max + 0.5 * dlat_out, dlat_out, dtype=np.float64)

    # ---- Wavelet projection ------------------------------------------------
    proj_result = _project_field_to_bmaux(
        eps_clean, t_eps, lon_r, lat_r, valid_mask,
        lon_out, lat_out,
        lmin, lmax, facpsp, npsp, ntheta,
        tdecmin, tdecmax, valid_min, verbose,
        _tag="calibrate_bmaux_dyn",
        tdec_bins=tdec_bins,
        n_jobs=n_jobs,
    )
    if tdec_bins is not None:
        ff, Std_ms, tdec_out = proj_result
    else:
        ff, Std, Tdec = proj_result

    # ---- Convert output lon to requested unit ------------------------------
    if lon_unit_out == "0_360":
        lon_out_save = lon_out % 360
    else:
        lon_out_save = ((lon_out + 180) % 360) - 180
    order = np.argsort(lon_out_save)
    lon_out_save = lon_out_save[order]

    attrs = {
        "description": "BMaux Std/Tdec calibrated from QG1L model-error forcing residuals.",
        "mode": "qg_residual",
        "lmin_km": lmin, "lmax_km": lmax,
        "facpsp": facpsp, "npsp": npsp, "ntheta": ntheta,
        "tdecmin": tdecmin, "tdecmax": tdecmax,
        "dlon_out": dlon_out, "dlat_out": dlat_out,
        "c0_ms": c_val,
        "dt_internal_s": dt_internal,
        "time_scheme": time_scheme,
        "forecast_steps": forecast_steps,
        "mdt_used": path_mdt if path_mdt else "none",
        "source_files": ";".join(files),
    }

    if tdec_bins is not None:
        Std_ms = Std_ms[:, :, :, order]
        out = xr.Dataset(
            data_vars={"Std": (("f", "tdec", "lat", "lon"), Std_ms)},
            coords={
                "f": ("f", ff.astype(np.float32)),
                "tdec": ("tdec", tdec_out.astype(np.float32)),
                "lat": ("lat", lat_out.astype(np.float32)),
                "lon": ("lon", lon_out_save.astype(np.float32)),
            },
            attrs=attrs,
        )
        out["f"].attrs["units"] = "1/km"
        out["tdec"].attrs["units"] = "days"
        out["tdec"].attrs["long_name"] = "decorrelation time bin centre"
        out["Std"].attrs["units"] = "m/day"
        out["Std"].encoding["dtype"] = "float32"
    else:
        Std = Std[:, :, order]
        Tdec = Tdec[:, :, order]
        out = xr.Dataset(
            data_vars={
                "Std": (("f", "lat", "lon"), Std),
                "Tdec": (("f", "lat", "lon"), Tdec),
            },
            coords={
                "f": ("f", ff.astype(np.float32)),
                "lat": ("lat", lat_out.astype(np.float32)),
                "lon": ("lon", lon_out_save.astype(np.float32)),
            },
            attrs=attrs,
        )
        out["f"].attrs["units"] = "1/km"
        out["Std"].attrs["units"] = "m/day"
        out["Std"].attrs["long_name"] = "wavelet-band QG forcing amplitude (m/day)"
        out["Tdec"].attrs["long_name"] = "decorrelation time in days"
        # Avoid CF timedelta auto-decoding on reload
        out["Std"].encoding["dtype"] = "float32"
        out["Tdec"].encoding["dtype"] = "float32"

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    out.to_netcdf(out_path)
    if verbose:
        print(f"[calibrate_bmaux_dyn] wrote {out_path}")
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate BMaux Std/Tdec from QG1L model-error forcing residuals."
    )
    # --- reference SSH ---
    p.add_argument("--ref", default=None, help="Reference SSH NetCDF path or glob.")
    p.add_argument("--out", required=True, help="Output NetCDF path.")
    p.add_argument(
        "--out-eps", default=None, metavar="PATH",
        help="If set, save raw QG residuals (m/day) to this NetCDF file.",
    )
    p.add_argument(
        "--in-eps", default=None, metavar="PATH",
        help=(
            "Load pre-computed QG residuals from this NetCDF file (written by a "
            "previous run with --out-eps) and skip reference-SSH loading and QGM "
            "computation entirely.  When set, --ref is not required."
        ),
    )
    p.add_argument("--name-time", default="time")
    p.add_argument("--name-lon", default="lon")
    p.add_argument("--name-lat", default="lat")
    p.add_argument("--name-ssh", default="ssh")
    # --- region ---
    p.add_argument("--lon-min", type=float, default=None)
    p.add_argument("--lon-max", type=float, default=None)
    p.add_argument("--lat-min", type=float, default=None)
    p.add_argument("--lat-max", type=float, default=None)
    # --- wavelet bands ---
    p.add_argument("--lmin", type=float, default=80.0)
    p.add_argument("--lmax", type=float, default=1000.0)
    p.add_argument("--facpsp", type=float, default=1.5)
    p.add_argument("--npsp", type=float, default=3.5)
    p.add_argument("--ntheta", type=int, default=4)
    # --- output grid ---
    p.add_argument("--dlon-out", type=float, default=1.0)
    p.add_argument("--dlat-out", type=float, default=1.0)
    # --- Tdec clamp ---
    p.add_argument("--tdecmin", type=float, default=2.5)
    p.add_argument("--tdecmax", type=float, default=40.0)
    p.add_argument("--valid-min", type=float, default=0.5)
    p.add_argument(
        "--forecast-days", type=float, default=None, metavar="DAYS",
        help=(
            "Forecast horizon in days for computing QG residuals (default: 1 reference "
            "time step). Longer horizons let QG errors accumulate and yield larger Std."
        ),
    )
    p.add_argument(
        "--n-jobs", type=int, default=1, metavar="N",
        help=(
            "Number of parallel worker processes for the wavelet projection "
            "(default 1 = serial). Use -1 for all available CPUs."
        ),
    )
    # --- QG model ---
    p.add_argument(
        "--dt-internal", type=float, default=1200.0,
        help="Internal QG time step (seconds). Default 1200 s (CFL-safe at ~5 km).",
    )
    p.add_argument(
        "--time-scheme", default="rk3", choices=["Euler", "rk2", "rk3"],
        help="QG time-integration scheme.",
    )
    p.add_argument(
        "--c0", type=float, default=2.7,
        help="First-baroclinic phase speed m/s (used if --filec not supplied).",
    )
    p.add_argument(
        "--filec", default=None,
        help="NetCDF with 2D phase-speed field (default: uniform c0).",
    )
    p.add_argument("--name-c-lon", default="lon")
    p.add_argument("--name-c-lat", default="lat")
    p.add_argument("--name-c-var", default="c1",
                   help="Variable name for c in --filec (default: c1).")
    p.add_argument(
        "--path-mdt", default=None,
        help="MDT NetCDF file. When set, Qgm runs on full SSH (MDT+SLA).",
    )
    p.add_argument("--name-mdt-lon", default="longitude")
    p.add_argument("--name-mdt-lat", default="latitude")
    p.add_argument("--name-mdt-var", default="mdt")
    p.add_argument(
        "--path-hb", default=None,
        help=(
            "NetCDF file with a prescribed background SSH field to use as hb "
            "(boundary / background state) in every Qgm.step call. "
            "Typical use: time-mean SSH or analysis SSH from a previous experiment. "
            "Default: zeros (pure SLA-anomaly mode)."
        ),
    )
    p.add_argument("--name-hb-time", default="time")
    p.add_argument("--name-hb-lon", default="lon")
    p.add_argument("--name-hb-lat", default="lat")
    p.add_argument("--name-hb-var", default="ssh",
                   help="Variable name for hb in --path-hb (default: ssh).")
    p.add_argument(
        "--min-hb-valid-frac", type=float, default=0.0, metavar="FRAC",
        help=(
            "Minimum fraction [0,1] of ocean points that must be valid (non-NaN) in a "
            "hb slice for it to be used. Steps below this fall back to ssh[n] + zeros "
            "(default: 0.0 = no filter)."
        ),
    )
    p.add_argument(
        "--regrid-qg", action="store_true",
        help=(
            "Interpolate the reference SSH to a uniform dx≈dy grid before running the QG "
            "model (dlon = dlat/cos(lat_mean)), then project residuals back to the original "
            "grid.  Recommended for domains spanning several degrees of latitude (default: off)."
        ),
    )
    p.add_argument(
        "--lon-unit-out", choices=["0_360", "-180_180"], default="0_360",
    )
    p.add_argument(
        "--tdec-bins", type=float, nargs="+", default=None, metavar="DAYS",
        help=(
            "Multi-scale mode: one or more decorrelation time bin centres (days). "
            "When supplied, output has Std(f,tdec,lat,lon) instead of Std+Tdec."
        ),
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    a = _parse_args(argv)
    if a.ref is None and a.in_eps is None:
        import sys as _sys
        _sys.exit("error: --ref is required unless --in-eps is provided")
    calibrate_bmaux_dyn(
        ref_path=a.ref,
        out_path=a.out,
        name_var={
            "time": a.name_time,
            "lon": a.name_lon,
            "lat": a.name_lat,
            "ssh": a.name_ssh,
        },
        lon_min=a.lon_min, lon_max=a.lon_max,
        lat_min=a.lat_min, lat_max=a.lat_max,
        lmin=a.lmin, lmax=a.lmax, facpsp=a.facpsp, npsp=a.npsp,
        ntheta=a.ntheta,
        dlon_out=a.dlon_out, dlat_out=a.dlat_out,
        tdecmin=a.tdecmin, tdecmax=a.tdecmax,
        valid_min=a.valid_min,
        dt_internal=a.dt_internal,
        time_scheme=a.time_scheme,
        c0=a.c0,
        filec=a.filec,
        name_var_c={
            "lon": a.name_c_lon,
            "lat": a.name_c_lat,
            "var": a.name_c_var,
        },
        path_mdt=a.path_mdt,
        name_var_mdt={
            "lon": a.name_mdt_lon,
            "lat": a.name_mdt_lat,
            "var": a.name_mdt_var,
        },
        lon_unit_out=a.lon_unit_out,
        forecast_days=a.forecast_days,
        out_eps_path=a.out_eps,
        in_eps_path=a.in_eps,
        path_hb=a.path_hb,
        name_var_hb={
            "time": a.name_hb_time,
            "lon": a.name_hb_lon,
            "lat": a.name_hb_lat,
            "var": a.name_hb_var,
        },
        min_hb_valid_frac=a.min_hb_valid_frac,
        regrid_qg=a.regrid_qg,
        n_jobs=a.n_jobs,
        tdec_bins=a.tdec_bins,
        verbose=not a.quiet,
    )


if __name__ == "__main__":
    main()
