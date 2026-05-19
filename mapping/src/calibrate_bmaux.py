#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline calibration of the BMaux auxiliary NetCDF file (Std, Tdec).

Projects a reference SSH field onto the SAME anisotropic Morlet-like wavelets
used by `BASIS_BMaux` (see mapping/src/basis.py, set_basis), then measures:
  - Std  : standard deviation of the wavelet-band envelope amplitude (m)
  - Tdec : 1/e decorrelation time of that envelope (days)

Per band of central wavenumber f (1/km) and per coarse output cell (lon, lat):
    DX  = npsp / (2 * f)                         # wavelet extent (km)
    facs(x,y) = mywindow(x/DX) * mywindow(y/DX)  # spatial taper, cos^2 lobe
    For each orientation theta in [0, pi):
        kx = 2*pi*f*cos(theta);  ky = 2*pi*f*sin(theta)
        Cc_th(t) = sum(facs * ssh(t,.) * cos(kx*x+ky*y)) / sum(facs^2 * cos^2)
        Cs_th(t) = sum(facs * ssh(t,.) * sin(kx*x+ky*y)) / sum(facs^2 * sin^2)
    Std  = sqrt( mean_theta ( Var_t(Cc_th) + Var_t(Cs_th) ) / 2 )
    Tdec = first lag (days) where the averaged ACF of (Cc_th, Cs_th) drops
           below 1/e, clamped to [tdecmin, tdecmax].

The output NetCDF schema matches `mapping/aux/aux_reduced_basis_BM.nc`:
    coords: f (1/km), lon (deg), lat (deg)
    vars  : Std(f, lat, lon), Tdec(f, lat, lon)  (float32)
This file can be plugged directly into BASIS_BMaux via `file_aux=...`.

Usage (as a script):
    python -m mapping.src.calibrate_bmaux \
        --ref /path/to/eNATL60_ssh.nc --name-ssh sossheig \
        --lon-min 294 --lon-max 306 --lat-min 32 --lat-max 44 \
        --lmin 80 --lmax 1000 --facpsp 1.5 --npsp 3.5 \
        --ntheta 4 --dlon-out 1.0 --dlat-out 1.0 \
        --out aux_bmaux_gulfstream.nc
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Optional, Sequence

import numpy as np
import xarray as xr


# ----------------------------------------------------------------------------
# Wavelet shape: must match basis.py:mywindow
# ----------------------------------------------------------------------------
def _mywindow(x: np.ndarray) -> np.ndarray:
    """cos^2 taper on [-1,1], 0 outside (matches basis.py:mywindow with clipping)."""
    y = np.where(np.abs(x) <= 1.0, np.cos(x * 0.5 * np.pi) ** 2, 0.0)
    return y


# ----------------------------------------------------------------------------
# Band frequencies: must match basis.py BasisBMaux.set_basis
# ----------------------------------------------------------------------------
def _band_frequencies(lmin: float, lmax: float, facpsp: float, npsp: float) -> np.ndarray:
    logff = np.arange(
        np.log(1.0 / lmin),
        np.log(1.0 / lmax) - np.log(1.0 + facpsp / npsp),
        -np.log(1.0 + facpsp / npsp),
    )[::-1]
    return np.exp(logff)


# ----------------------------------------------------------------------------
# Decorrelation time from FFT-based autocorrelation
# ----------------------------------------------------------------------------
def _acf_normalized(a: np.ndarray) -> Optional[np.ndarray]:
    """Return normalized autocorrelation of `a` (length n), or None if invalid."""
    a = a - np.nanmean(a)
    n = a.size
    if n < 4 or not np.isfinite(a).any() or np.nanstd(a) == 0:
        return None
    a = np.where(np.isfinite(a), a, 0.0)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(a, n=nfft)
    acf = np.fft.irfft(F * np.conj(F), n=nfft)[:n]
    if acf[0] <= 0:
        return None
    return acf / acf[0]


def _tdec_from_acf(acf: np.ndarray, dt_days: float) -> float:
    """First lag (days) where `acf` drops below 1/e (linearly interpolated)."""
    thr = 1.0 / np.e
    n = acf.size
    below = np.where(acf < thr)[0]
    if below.size == 0:
        return n * dt_days
    k = int(below[0])
    if k == 0:
        return 0.0
    y0, y1 = acf[k - 1], acf[k]
    frac = (y0 - thr) / (y0 - y1) if y0 != y1 else 0.0
    return (k - 1 + frac) * dt_days


# ----------------------------------------------------------------------------
# Multi-scale spectral partitioning
# ----------------------------------------------------------------------------
def _temporal_bandpass_stds(
    series: np.ndarray,
    dt_days: float,
    tdec_bins: np.ndarray,
) -> np.ndarray:
    """Partition temporal variance of ``series`` across decorrelation-time bins.

    Uses FFT spectral partitioning.  Bin boundaries are log-geometric midpoints
    between adjacent bin centres.  Returned values satisfy:

        sum_k  Std_k**2  ≈  Var(series) / 2  ≈  single-scale Std**2

    so there is no double-counting across bins.

    Parameters
    ----------
    series : (nt,) float
    dt_days : float
        Sampling interval in days.
    tdec_bins : (n_bins,) float
        Target decorrelation times (days).  Any order; results are returned
        in the same order as ``tdec_bins``.

    Returns
    -------
    Std_k : (n_bins,) float
    """
    tdec_bins = np.asarray(tdec_bins, dtype=float)
    n_bins = len(tdec_bins)
    N = len(series)
    if N < 4 or n_bins == 0:
        return np.zeros(n_bins)

    # Sort bins by ascending frequency (= descending Tdec) to build contiguous bands
    freq_centers = 1.0 / tdec_bins          # 1/day
    sort_idx = np.argsort(freq_centers)     # ascending freq
    freq_s = freq_centers[sort_idx]

    zm = series - np.mean(series)
    F = np.fft.rfft(zm)                     # shape (N//2 + 1,)
    freqs = np.fft.rfftfreq(N, d=dt_days)   # 1/day, >= 0
    power = np.abs(F) ** 2                  # |X[k]|^2

    # Bin boundary frequencies: geometric midpoints between adjacent centres
    f_lo = np.empty(n_bins)
    f_hi = np.empty(n_bins)
    for k in range(n_bins):
        f_lo[k] = 0.0 if k == 0 else np.sqrt(freq_s[k - 1] * freq_s[k])
        f_hi[k] = np.inf if k == n_bins - 1 else np.sqrt(freq_s[k] * freq_s[k + 1])

    # Nyquist frequency (should not be doubled in the one-sided factor)
    nyq = freqs[-1] if (N % 2 == 0) else None

    Std_s = np.zeros(n_bins)
    for k in range(n_bins):
        mask = (freqs > f_lo[k]) & (freqs <= f_hi[k]) & (freqs > 0.0)
        if not mask.any():
            continue
        f_in = freqs[mask]
        p_in = power[mask]
        # Factor 2 for one-sided rfft (negative freqs not included),
        # except for the Nyquist which appears once.
        if nyq is not None and np.any(f_in == nyq):
            nyq_mask = f_in == nyq
            var_k = (2.0 * np.sum(p_in[~nyq_mask]) + np.sum(p_in[nyq_mask])) / N ** 2
        else:
            var_k = 2.0 * np.sum(p_in) / N ** 2
        Std_s[k] = np.sqrt(max(0.0, var_k) / 2.0)

    # Reorder back to original tdec_bins order
    Std_out = np.empty(n_bins)
    Std_out[sort_idx] = Std_s
    return Std_out


# ----------------------------------------------------------------------------
# Band-parallel projection helpers (module-level so they are picklable)
# ----------------------------------------------------------------------------
_proj_band_state: dict = {}


def _init_proj_worker(state: dict) -> None:
    """Spawn-safe worker initializer: populate module-level state in each worker process."""
    _proj_band_state.update(state)


def _project_one_band(iff):
    """Project field onto one frequency band. Reads shared data from _proj_band_state."""
    bs = _proj_band_state
    ff = bs["ff"]; DX = bs["DX"]; field = bs["field"]
    dt_days = bs["dt_days"]
    lon_r = bs["lon_r"]; lat_r = bs["lat_r"]; valid_mask = bs["valid_mask"]
    lon_out = bs["lon_out"]; lat_out = bs["lat_out"]
    ntheta = bs["ntheta"]; tdecmin = bs["tdecmin"]; tdecmax = bs["tdecmax"]
    valid_min = bs["valid_min"]; tdec_bins_arr = bs["tdec_bins_arr"]
    cos_th = bs["cos_th"]; sin_th = bs["sin_th"]
    Y = bs["Y"]; cos_lat_out = bs["cos_lat_out"]

    f = ff[iff]
    dx_band = DX[iff]
    k = 2.0 * np.pi * f
    km2deg = 1.0 / 110.0
    nlon_out = lon_out.size
    nlat_out = lat_out.size
    _n_tdec = len(tdec_bins_arr) if tdec_bins_arr is not None else 0

    std_band = np.full((nlat_out, nlon_out), np.nan, dtype=np.float32)
    tdec_band = np.full((nlat_out, nlon_out), np.nan, dtype=np.float32)
    std_ms_band = (
        np.full((_n_tdec, nlat_out, nlon_out), np.nan, dtype=np.float32)
        if _n_tdec > 0 else None
    )

    for jo in range(nlat_out):
        y_col = Y[:, jo]
        j_in = np.where(np.abs(y_col) <= dx_band)[0]
        if j_in.size == 0:
            continue
        y_sub = y_col[j_in]
        wy = _mywindow(y_sub / dx_band)
        for io in range(nlon_out):
            x_row = (lon_r - lon_out[io]) / km2deg * cos_lat_out[jo]
            x_row = np.where(x_row > 180.0 / km2deg, x_row - 360.0 / km2deg, x_row)
            x_row = np.where(x_row < -180.0 / km2deg, x_row + 360.0 / km2deg, x_row)
            i_in = np.where(np.abs(x_row) <= dx_band)[0]
            if i_in.size == 0:
                continue
            x_sub = x_row[i_in]
            wx = _mywindow(x_sub / dx_band)
            facs_full = wy[:, None] * wx[None, :]
            vmask = valid_mask[np.ix_(j_in, i_in)]
            facs = facs_full * vmask
            full_mass = float((facs_full ** 2).sum())
            if full_mass <= 0:
                continue
            coverage = float((facs ** 2).sum()) / full_mass
            if coverage < valid_min:
                continue
            if facs.sum() <= 0:
                continue
            xx2 = np.broadcast_to(x_sub[None, :], facs.shape)
            yy2 = np.broadcast_to(y_sub[:, None], facs.shape)
            field_sub = field[:, j_in[:, None], i_in[None, :]]
            var_sum = 0.0
            acf_sum = None
            acf_count = 0
            ok = 0
            Std_k_sum = np.zeros(_n_tdec) if tdec_bins_arr is not None else None
            for it in range(ntheta):
                phase = k * (cos_th[it] * xx2 + sin_th[it] * yy2)
                c = np.cos(phase)
                s = np.sin(phase)
                nC = float(np.sum(facs * facs * c * c))
                nS = float(np.sum(facs * facs * s * s))
                if nC <= 0 or nS <= 0:
                    continue
                fc = facs * c
                fs = facs * s
                Cc = np.einsum("ji,tji->t", fc, field_sub) / nC
                Cs = np.einsum("ji,tji->t", fs, field_sub) / nS
                var_sum += float(np.var(Cc) + np.var(Cs))
                if tdec_bins_arr is not None:
                    Std_k_sum += _temporal_bandpass_stds(Cc, dt_days, tdec_bins_arr) ** 2
                    Std_k_sum += _temporal_bandpass_stds(Cs, dt_days, tdec_bins_arr) ** 2
                for series in (Cc, Cs):
                    acf_n = _acf_normalized(series)
                    if acf_n is not None:
                        if acf_sum is None:
                            acf_sum = acf_n.copy()
                            acf_count = 1
                        else:
                            acf_sum += acf_n
                            acf_count += 1
                ok += 1
            if ok == 0:
                continue
            std_band[jo, io] = np.float32(np.sqrt(0.5 * var_sum / ok))
            if tdec_bins_arr is not None:
                std_ms_band[:, jo, io] = np.sqrt(0.5 * Std_k_sum / ok).astype(np.float32)
            if acf_sum is not None and acf_count > 0:
                td = _tdec_from_acf(acf_sum / acf_count, dt_days)
                td = float(np.clip(td, tdecmin, tdecmax)) if np.isfinite(td) else float(tdecmin)
            else:
                td = float(tdecmin)
            tdec_band[jo, io] = np.float32(td)

    return iff, std_band, tdec_band, std_ms_band


# ----------------------------------------------------------------------------
# Shared wavelet projection (used by both calibrate_bmaux and calibrate_bmaux_dyn)
# ----------------------------------------------------------------------------
def _project_field_to_bmaux(
    field: np.ndarray,
    t_days: np.ndarray,
    lon_r: np.ndarray,
    lat_r: np.ndarray,
    valid_mask: np.ndarray,
    lon_out: np.ndarray,
    lat_out: np.ndarray,
    lmin: float,
    lmax: float,
    facpsp: float,
    npsp: float,
    ntheta: int,
    tdecmin: float,
    tdecmax: float,
    valid_min: float = 0.5,
    verbose: bool = True,
    _tag: str = "project",
    tdec_bins=None,
    n_jobs: int = 1,
) -> tuple:
    """Project a 3D field onto BasisBMaux wavelet bands → Std, Tdec.

    Parameters
    ----------
    field : (nt, ny, nx) float64
        Field to project (NaN already replaced with 0; valid_mask tracks ocean).
        Units are arbitrary (m for SSH, m/day for QG residuals, …).
    t_days : (nt,) float
        Time axis in days (uniform spacing assumed).
    lon_r, lat_r : 1D float arrays
        Reference grid coordinate axes (degrees).
    valid_mask : (ny, nx) bool
        True where field is valid (ocean).
    lon_out, lat_out : 1D float arrays
        Coarse output grid centres (degrees, in the same lon convention as lon_r).
    lmin, lmax, facpsp, npsp : float
        Wavelet band definition — must match BasisBMaux config.
    ntheta : int
        Number of orientations averaged in [0, π).
    tdecmin, tdecmax : float
        Clamp Tdec to this interval (days).
    valid_min : float
        Minimum ocean window-mass fraction (0–1) for a coarse cell.
    verbose : bool
    _tag : str
        Prefix for log messages.

    Returns
    -------
    ff : (nf,) float64
        Band central frequencies (1/km).
    Std : (nf, nlat_out, nlon_out) float32
    Tdec : (nf, nlat_out, nlon_out) float32
    """
    nt = t_days.size
    dt_days = float(np.median(np.diff(t_days))) if nt > 1 else 1.0
    nlon_out = lon_out.size
    nlat_out = lat_out.size

    # ---- Band frequencies & orientations -----------------------------------
    ff = _band_frequencies(lmin, lmax, facpsp, npsp)
    nf = ff.size
    DX = 0.5 * npsp / ff  # km
    theta = np.pi * np.arange(ntheta) / ntheta
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    if verbose:
        print(f"[{_tag}] nf={nf}, ntheta={ntheta}")
        print(f"[{_tag}] wavelengths (km): {1.0/ff}")

    # ---- Precompute geometry (km from each output centre) ------------------
    km2deg = 1.0 / 110.0
    Y = (lat_r[:, None] - lat_out[None, :]) / km2deg  # (ny, nlat_out) km
    cos_lat_out = np.cos(lat_out * np.pi / 180.0)

    Std = np.full((nf, nlat_out, nlon_out), np.nan, dtype=np.float32)
    Tdec = np.full((nf, nlat_out, nlon_out), np.nan, dtype=np.float32)

    if tdec_bins is not None:
        _tdec_arr = np.sort(np.asarray(tdec_bins, dtype=float))
        _n_tdec = len(_tdec_arr)
        Std_ms = np.full((nf, _n_tdec, nlat_out, nlon_out), np.nan, dtype=np.float32)
    else:
        _tdec_arr = None
        _n_tdec = 0
        Std_ms = None

    # ---- Loop bands (parallel or serial) -----------------------------------
    if n_jobs != 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        actual_jobs = min(n_jobs if n_jobs > 0 else (os.cpu_count() or 1), nf)
        if verbose:
            print(f"[{_tag}] parallelising {nf} bands over {actual_jobs} processes")
        _state_snapshot = dict(
            ff=ff, DX=DX, field=field, dt_days=dt_days,
            lon_r=lon_r, lat_r=lat_r, valid_mask=valid_mask,
            lon_out=lon_out, lat_out=lat_out,
            ntheta=ntheta, tdecmin=tdecmin, tdecmax=tdecmax,
            valid_min=valid_min, tdec_bins_arr=_tdec_arr,
            cos_th=cos_th, sin_th=sin_th, Y=Y, cos_lat_out=cos_lat_out,
        )
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=actual_jobs, mp_context=ctx,
            initializer=_init_proj_worker, initargs=(_state_snapshot,),
        ) as pool:
            for iff_r, std_b, tdec_b, std_ms_b in pool.map(_project_one_band, range(nf)):
                if verbose:
                    print(f"  band {iff_r+1}/{nf}: lambda={1.0/ff[iff_r]:.1f} km  [done]")
                Std[iff_r] = std_b
                Tdec[iff_r] = tdec_b
                if tdec_bins is not None and std_ms_b is not None:
                    Std_ms[iff_r] = std_ms_b
    else:
        for iff in range(nf):
            f = ff[iff]
            dx_band = DX[iff]
            k = 2.0 * np.pi * f
            if verbose:
                print(f"  band {iff+1}/{nf}: lambda={1.0/f:.1f} km, DX={dx_band:.1f} km")

            for jo in range(nlat_out):
                y_col = Y[:, jo]  # (ny,)
                j_in = np.where(np.abs(y_col) <= dx_band)[0]
                if j_in.size == 0:
                    continue
                y_sub = y_col[j_in]
                wy = _mywindow(y_sub / dx_band)
                for io in range(nlon_out):
                    x_row = (lon_r - lon_out[io]) / km2deg * cos_lat_out[jo]
                    # wrap-around for 0/360 boundary
                    x_row = np.where(x_row > 180.0 / km2deg, x_row - 360.0 / km2deg, x_row)
                    x_row = np.where(x_row < -180.0 / km2deg, x_row + 360.0 / km2deg, x_row)
                    i_in = np.where(np.abs(x_row) <= dx_band)[0]
                    if i_in.size == 0:
                        continue
                    x_sub = x_row[i_in]
                    wx = _mywindow(x_sub / dx_band)
                    # 2D taper and validity
                    facs_full = wy[:, None] * wx[None, :]  # (nj, ni)
                    vmask = valid_mask[np.ix_(j_in, i_in)]
                    facs = facs_full * vmask
                    full_mass = float((facs_full ** 2).sum())
                    if full_mass <= 0:
                        continue
                    coverage = float((facs ** 2).sum()) / full_mass
                    if coverage < valid_min:
                        continue
                    if facs.sum() <= 0:
                        continue
                    xx2 = np.broadcast_to(x_sub[None, :], facs.shape)
                    yy2 = np.broadcast_to(y_sub[:, None], facs.shape)
                    field_sub = field[:, j_in[:, None], i_in[None, :]]  # (nt, nj, ni)
                    var_sum = 0.0
                    acf_sum = None
                    acf_count = 0
                    ok = 0
                    Std_k_sum = np.zeros(_n_tdec) if tdec_bins is not None else None
                    for it in range(ntheta):
                        phase = k * (cos_th[it] * xx2 + sin_th[it] * yy2)
                        c = np.cos(phase)
                        s = np.sin(phase)
                        nC = float(np.sum(facs * facs * c * c))
                        nS = float(np.sum(facs * facs * s * s))
                        if nC <= 0 or nS <= 0:
                            continue
                        fc = facs * c
                        fs = facs * s
                        Cc = np.einsum("ji,tji->t", fc, field_sub) / nC
                        Cs = np.einsum("ji,tji->t", fs, field_sub) / nS
                        var_sum += float(np.var(Cc) + np.var(Cs))
                        if tdec_bins is not None:
                            Std_k_sum += _temporal_bandpass_stds(Cc, dt_days, _tdec_arr) ** 2
                            Std_k_sum += _temporal_bandpass_stds(Cs, dt_days, _tdec_arr) ** 2
                        for series in (Cc, Cs):
                            acf_n = _acf_normalized(series)
                            if acf_n is not None:
                                if acf_sum is None:
                                    acf_sum = acf_n.copy()
                                    acf_count = 1
                                else:
                                    acf_sum += acf_n
                                    acf_count += 1
                        ok += 1
                    if ok == 0:
                        continue
                    Std[iff, jo, io] = np.float32(np.sqrt(0.5 * var_sum / ok))
                    if tdec_bins is not None:
                        Std_ms[iff, :, jo, io] = np.sqrt(0.5 * Std_k_sum / ok).astype(np.float32)
                    if acf_sum is not None and acf_count > 0:
                        td = _tdec_from_acf(acf_sum / acf_count, dt_days)
                        td = float(np.clip(td, tdecmin, tdecmax)) if np.isfinite(td) else float(tdecmin)
                    else:
                        td = float(tdecmin)
                    Tdec[iff, jo, io] = np.float32(td)

    if tdec_bins is not None:
        return ff, Std_ms, _tdec_arr
    return ff, Std, Tdec


# ----------------------------------------------------------------------------
# Core calibration
# ----------------------------------------------------------------------------
def calibrate_bmaux(
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
    time_chunk: Optional[int] = None,
    lon_unit_out: str = "0_360",
    verbose: bool = True,
    tdec_bins=None,
) -> xr.Dataset:
    """Calibrate Std/Tdec aux file from a reference SSH dataset.

    Parameters
    ----------
    ref_path : str or list of str
        Path (or glob, or list) to NetCDF reference SSH file(s).
    out_path : str
        Output NetCDF path.
    name_var : dict, optional
        {'time': ..., 'lon': ..., 'lat': ..., 'ssh': ...}.
        Defaults to {'time':'time','lon':'lon','lat':'lat','ssh':'ssh'}.
    lon_min, lon_max, lat_min, lat_max : float, optional
        Calibration region. Defaults to full dataset extent.
    lmin, lmax, facpsp, npsp : float
        Wavelet band definition (must match BASIS_BMaux config).
    ntheta : int
        Number of orientations averaged in [0, pi).
    dlon_out, dlat_out : float
        Output map grid spacing (deg).
    tdecmin, tdecmax : float
        Clamp Tdec to this interval (days).
    lon_unit_out : str
        '0_360' (default, matches shipped aux file) or '-180_180'.
    """
    if name_var is None:
        name_var = {"time": "time", "lon": "lon", "lat": "lat", "ssh": "ssh"}
    nm_t = name_var["time"]
    nm_lon = name_var["lon"]
    nm_lat = name_var["lat"]
    nm_ssh = name_var["ssh"]

    # ---- Load reference -----------------------------------------------------
    if isinstance(ref_path, (list, tuple)):
        files = list(ref_path)
    else:
        files = sorted(glob.glob(ref_path)) if any(c in str(ref_path) for c in "*?[") else [ref_path]
    if not files:
        raise FileNotFoundError(f"No reference files match: {ref_path}")
    if verbose:
        print(f"[calibrate_bmaux] opening {len(files)} reference file(s)")
    open_kwargs = {"combine": "by_coords"} if len(files) > 1 else {}
    if len(files) > 1:
        ds = xr.open_mfdataset(files, **open_kwargs)
    else:
        ds = xr.open_dataset(files[0])

    # Coordinates
    lon = ds[nm_lon].values
    lat = ds[nm_lat].values
    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("Reference lon/lat must be 1D (regular grid).")

    # Time -> days since first record
    t_da = ds[nm_t]
    t_vals = t_da.values
    if np.issubdtype(t_vals.dtype, np.datetime64):
        t_sec = (t_vals - t_vals[0]) / np.timedelta64(1, "s")
    else:
        # assume already numeric; try units attr
        units = t_da.attrs.get("units", "seconds since 1970-01-01")
        # fall back: treat as seconds
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
    if verbose:
        print(f"[calibrate_bmaux] {nt} time steps, dt = {dt_days:.3f} days")

    # ---- Region crop -------------------------------------------------------
    # Detect lon unit of reference
    ref_lon_unit = "0_360" if np.nanmin(lon) >= 0 and np.nanmax(lon) > 180 else "-180_180"
    if lon_min is None:
        lon_min = float(np.nanmin(lon))
    if lon_max is None:
        lon_max = float(np.nanmax(lon))
    if lat_min is None:
        lat_min = float(np.nanmin(lat))
    if lat_max is None:
        lat_max = float(np.nanmax(lat))

    # Bring region bounds into reference lon unit
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
        raise ValueError("Region crop too small (need >=4 pixels in each direction).")
    lon_r = lon[i_sel]
    lat_r = lat[j_sel]

    # Load SSH cropped (keep in memory; calibration is per-band so we re-use it)
    ssh = ds[nm_ssh].isel({nm_lat: j_sel, nm_lon: i_sel}).values.astype(np.float64)
    # Expect shape (nt, ny, nx); transpose if needed
    if ssh.ndim != 3:
        raise ValueError(f"SSH must be 3D (time,lat,lon), got shape {ssh.shape}")
    if ssh.shape != (nt, lat_r.size, lon_r.size):
        # Try common alternative orderings
        dim_order = ds[nm_ssh].dims
        order = [dim_order.index(d) for d in (nm_t, nm_lat, nm_lon)]
        ssh = np.transpose(ssh, order)
    valid_mask = np.isfinite(ssh[0])
    ssh = np.where(np.isfinite(ssh), ssh, 0.0)
    if verbose:
        print(f"[calibrate_bmaux] SSH cropped to ({nt}, {lat_r.size}, {lon_r.size})")

    # ---- Output coarse grid -------------------------------------------------
    lon_out = np.arange(lon_min, lon_max + 0.5 * dlon_out, dlon_out, dtype=np.float64)
    lat_out = np.arange(lat_min, lat_max + 0.5 * dlat_out, dlat_out, dtype=np.float64)

    # ---- Wavelet projection -------------------------------------------------
    proj_result = _project_field_to_bmaux(
        ssh, t_days, lon_r, lat_r, valid_mask,
        lon_out, lat_out,
        lmin, lmax, facpsp, npsp, ntheta,
        tdecmin, tdecmax, valid_min, verbose,
        _tag="calibrate_bmaux",
        tdec_bins=tdec_bins,
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
        "description": "BMaux Std/Tdec calibrated from reference SSH (wavelet projection).",
        "lmin_km": lmin, "lmax_km": lmax,
        "facpsp": facpsp, "npsp": npsp, "ntheta": ntheta,
        "tdecmin": tdecmin, "tdecmax": tdecmax,
        "dlon_out": dlon_out, "dlat_out": dlat_out,
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
        out["Std"].attrs["units"] = "m"
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
        out["Std"].attrs["units"] = "m"
        out["Tdec"].attrs["long_name"] = "decorrelation time in days"
        # Avoid CF timedelta auto-decoding on reload (basis.py reads with
        # decode_times=False but xarray still decodes timedelta-like units).
        out["Std"].encoding["dtype"] = "float32"
        out["Tdec"].encoding["dtype"] = "float32"

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    out.to_netcdf(out_path)
    if verbose:
        print(f"[calibrate_bmaux] wrote {out_path}")
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate BMaux Std/Tdec from a reference SSH dataset.")
    p.add_argument("--ref", required=True, help="Reference SSH NetCDF path or glob.")
    p.add_argument("--out", required=True, help="Output NetCDF path.")
    p.add_argument("--name-time", default="time")
    p.add_argument("--name-lon", default="lon")
    p.add_argument("--name-lat", default="lat")
    p.add_argument("--name-ssh", default="ssh")
    p.add_argument("--lon-min", type=float, default=None)
    p.add_argument("--lon-max", type=float, default=None)
    p.add_argument("--lat-min", type=float, default=None)
    p.add_argument("--lat-max", type=float, default=None)
    p.add_argument("--lmin", type=float, default=80.0)
    p.add_argument("--lmax", type=float, default=1000.0)
    p.add_argument("--facpsp", type=float, default=1.5)
    p.add_argument("--npsp", type=float, default=3.5)
    p.add_argument("--ntheta", type=int, default=4)
    p.add_argument("--dlon-out", type=float, default=1.0)
    p.add_argument("--dlat-out", type=float, default=1.0)
    p.add_argument("--tdecmin", type=float, default=2.5)
    p.add_argument("--tdecmax", type=float, default=40.0)
    p.add_argument("--valid-min", type=float, default=0.5,
                   help="Minimum window mass coverage (0-1); cells below are skipped.")
    p.add_argument(
        "--tdec-bins", type=float, nargs="+", default=None, metavar="DAYS",
        help=(
            "Multi-scale mode: one or more decorrelation time bin centres (days). "
            "When supplied, output has Std(f,tdec,lat,lon) instead of Std+Tdec."
        ),
    )
    p.add_argument("--lon-unit-out", choices=["0_360", "-180_180"], default="0_360")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    a = _parse_args(argv)
    calibrate_bmaux(
        ref_path=a.ref,
        out_path=a.out,
        name_var={"time": a.name_time, "lon": a.name_lon, "lat": a.name_lat, "ssh": a.name_ssh},
        lon_min=a.lon_min, lon_max=a.lon_max,
        lat_min=a.lat_min, lat_max=a.lat_max,
        lmin=a.lmin, lmax=a.lmax, facpsp=a.facpsp, npsp=a.npsp,
        ntheta=a.ntheta,
        dlon_out=a.dlon_out, dlat_out=a.dlat_out,
        tdecmin=a.tdecmin, tdecmax=a.tdecmax,
        valid_min=a.valid_min,
        lon_unit_out=a.lon_unit_out,
        tdec_bins=a.tdec_bins,
        verbose=not a.quiet,
    )


if __name__ == "__main__":
    main()
