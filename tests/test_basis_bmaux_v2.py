"""Tests for Basis_bmaux_v2: self-consistent Q normalisation.

Three tests:
  1. Algebraic round-trip: _Q_from_std(sqrt(3*Q*f/(4*d²)), d) == Q exactly.
  2. Q_v2 > Q_legacy for typical ocean parameters (Std≪1, tdec≫1 days).
  3. Calibration round-trip: run calibrate_bmaux on a synthetic SSH field,
     then verify that _Q_from_std reproduces Std when the predicted temporal
     variance is back-computed via the calibrator formula.

Run as:
    python tests/test_basis_bmaux_v2.py
or:
    pytest tests/test_basis_bmaux_v2.py -v
"""
import os
import sys
import pathlib
import tempfile

import numpy as np
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mapping.src.basis import Basis_bmaux_v2
from mapping.src.calibrate_bmaux import calibrate_bmaux, _band_frequencies


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_v2_basis(facnlt: float) -> Basis_bmaux_v2:
    """Create a Basis_bmaux_v2 instance without calling __init__.

    Only self.facnlt is needed by _Q_from_std, so we bypass the full
    constructor (which requires config + State objects).
    """
    basis = Basis_bmaux_v2.__new__(Basis_bmaux_v2)
    basis.facnlt = facnlt
    return basis


# ---------------------------------------------------------------------------
# Test 1 – algebraic round-trip
# ---------------------------------------------------------------------------

def test_Q_from_std_round_trip():
    """_Q_from_std inverts the calibrator formula exactly.

    The calibrator produces Std_file = sqrt(3 * Q * facnlt / (4 * tdec²)).
    Basis_bmaux_v2._Q_from_std(Std_file, tdec) must return Q.
    """
    basis = _make_v2_basis(facnlt=2.0)

    for Q_true in [0.01, 0.1, 1.0, 10.0]:
        for tdec in [5.0, 10.0, 20.0]:
            Std_file = np.sqrt(3.0 * Q_true * basis.facnlt / (4.0 * tdec ** 2))
            Q_recovered = basis._Q_from_std(Std_file, tdec)
            rel_err = abs(Q_recovered / Q_true - 1.0)
            assert rel_err < 1e-10, (
                f"Round-trip failed: Q_true={Q_true}, tdec={tdec}, "
                f"Std_file={Std_file:.6g}, Q_recovered={Q_recovered:.6g}, "
                f"rel_err={rel_err:.2e}"
            )

    print("test_Q_from_std_round_trip: OK")


# ---------------------------------------------------------------------------
# Test 2 – formula value check
# ---------------------------------------------------------------------------

def test_Q_from_std_formula():
    """_Q_from_std returns exactly 4 * Std² * tdec² / (3 * facnlt).

    This is independent of whether Q_v2 is larger or smaller than Std;
    the formula is exact and dimensionally consistent.
    """
    for facnlt in [1.5, 2.0, 3.0]:
        basis = _make_v2_basis(facnlt=facnlt)
        for std_val in [0.05, 0.1, 0.2]:
            for tdec in [5.0, 10.0, 20.0]:
                Q_v2     = basis._Q_from_std(std_val, tdec)
                Q_expect = 4.0 * std_val ** 2 * tdec ** 2 / (3.0 * facnlt)
                assert abs(Q_v2 / Q_expect - 1.0) < 1e-10, (
                    f"Formula mismatch: facnlt={facnlt}, Std={std_val}, tdec={tdec}: "
                    f"got {Q_v2:.6g}, expected {Q_expect:.6g}"
                )

    print("test_Q_from_std_formula: OK")


# ---------------------------------------------------------------------------
# Test 3 – calibration then self-consistent Q
# ---------------------------------------------------------------------------

def test_calibration_then_Q_v2():
    """Calibrate a synthetic SSH, then check _Q_from_std self-consistency.

    Builds a plane-wave SSH with amplitude A0 and AR(1) decorrelation time
    tau0.  Runs calibrate_bmaux on the target wavelength band.  Then:
      (a) Verifies the algebraic round-trip:
              sqrt(3 * Q_v2 * facnlt / (4 * Tdec²)) == Std_measured.
      (b) Checks Q_v2 > Q_legacy = Std_measured.
      (c) Checks Q_v2 is plausible relative to the input amplitude A0
          (rough physical sanity: not off by orders of magnitude).
    """
    rng = np.random.default_rng(99)

    # --- domain -----------------------------------------------------------
    lat = np.arange(33.0, 43.0 + 1e-9, 0.1)
    lon = np.arange(295.0, 305.0 + 1e-9, 0.1)
    nlat, nlon = lat.size, lon.size
    nt = 200
    dt_days = 1.0
    t_days = np.arange(nt) * dt_days
    time = (np.array(t_days * 86400.0, dtype="timedelta64[s]")
            + np.datetime64("2012-10-01"))

    # --- target band -------------------------------------------------------
    ff = _band_frequencies(80.0, 1000.0, facpsp=1.5, npsp=3.5)
    ib = len(ff) // 2
    f0 = ff[ib]
    lam0 = 1.0 / f0

    # --- synthetic SSH = A0 * cos(2π f0 x) * a(t) -------------------------
    lat0_c = 38.0
    cos_lat = np.cos(lat0_c * np.pi / 180.0)
    x_km = (lon - 300.0) * 110.0 * cos_lat   # km, centred in domain
    A0 = 0.2    # m/day
    tau0 = 10.0  # days
    rho = np.exp(-dt_days / tau0)
    eps = rng.standard_normal(nt)
    a = np.empty(nt)
    a[0] = eps[0]
    for i in range(1, nt):
        a[i] = rho * a[i - 1] + np.sqrt(1.0 - rho ** 2) * eps[i]
    a = a / np.std(a) * A0

    ssh = (a[:, None, None] * np.cos(2.0 * np.pi * f0 * x_km)[None, None, :]
           * np.ones((1, nlat, 1)))

    ds = xr.Dataset(
        {"ssh": (("time", "lat", "lon"), ssh.astype("float32"))},
        coords={
            "time": time,
            "lat": lat.astype("float32"),
            "lon": lon.astype("float32"),
        },
    )

    facnlt = 2.0
    ntheta = 4

    with tempfile.TemporaryDirectory() as td:
        ref = os.path.join(td, "ref.nc")
        out = os.path.join(td, "aux.nc")
        ds.to_netcdf(ref)

        res = calibrate_bmaux(
            ref_path=ref, out_path=out,
            lon_min=297.0, lon_max=303.0, lat_min=35.0, lat_max=41.0,
            lmin=80.0, lmax=1000.0, facpsp=1.5, npsp=3.5,
            ntheta=ntheta, dlon_out=2.0, dlat_out=2.0,
            tdecmin=0.1, tdecmax=200.0, lon_unit_out="0_360",
            verbose=False,
        )

    Std_measured  = float(np.nanmean(res["Std"].isel(f=ib).values))
    Tdec_measured = float(np.nanmean(res["Tdec"].isel(f=ib).values))

    basis = _make_v2_basis(facnlt=facnlt)
    Q_v2    = basis._Q_from_std(Std_measured, Tdec_measured)
    Q_legacy = float(Std_measured)   # Basis_bmaux convention

    # (a) algebraic round-trip -----------------------------------------------
    Std_backpred = np.sqrt(3.0 * Q_v2 * facnlt / (4.0 * Tdec_measured ** 2))
    rel_err = abs(Std_backpred / Std_measured - 1.0)
    assert rel_err < 1e-10, (
        f"Round-trip failed: Std_in={Std_measured:.6f}, "
        f"Std_back={Std_backpred:.6f}, rel_err={rel_err:.2e}"
    )

    # (b) Q_v2 > Q_legacy ----------------------------------------------------
    assert Q_v2 > Q_legacy, (
        f"Expected Q_v2 > Q_legacy but Q_v2={Q_v2:.4f}, Q_legacy={Q_legacy:.4f}"
    )

    # (c) physical plausibility: Q_v2 within 2 decades of A0^2 --------------
    A0_sq = A0 ** 2
    assert 1e-2 * A0_sq < Q_v2 < 1e2 * A0_sq, (
        f"Q_v2={Q_v2:.4f} seems implausible relative to A0²={A0_sq:.4f}"
    )

    print(f"  lambda0 = {lam0:.1f} km")
    print(f"  Std_measured  = {Std_measured:.4f} m/day  (expect ~{A0/np.sqrt(2*ntheta):.4f})")
    print(f"  Tdec_measured = {Tdec_measured:.2f} d   (expect ~{tau0:.1f} d)")
    print(f"  Q_v2    = {Q_v2:.4f}  (legacy Q = {Q_legacy:.4f}, ratio = {Q_v2/Q_legacy:.1f}×)")
    print("test_calibration_then_Q_v2: OK")


# ---------------------------------------------------------------------------

def main():
    test_Q_from_std_round_trip()
    test_Q_from_std_formula()
    test_calibration_then_Q_v2()
    print("\nAll Basis_bmaux_v2 tests passed.")


if __name__ == "__main__":
    main()
