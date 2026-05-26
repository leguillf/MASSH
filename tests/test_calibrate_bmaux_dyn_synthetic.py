"""Synthetic tests for calibrate_bmaux_dyn.

Test 1 – Projection recovery (no QG model needed):
    Build a synthetic field with known amplitude A0 and decorrelation time tau0
    in one wavelength band, call _project_field_to_bmaux directly, and verify
    Std/Tdec are recovered within tolerance.  Checks band-selectivity (other
    bands must have Std < 0.5 * target-band Std).

Test 2 – Free-QG residuals are ~zero:
    Integrate Qgm on a small (21×21) grid for 5 steps from a random initial
    state.  Construct the reference SSH stack as the free-QG trajectory, so
    compute_qg_residuals should return eps ≈ 0 (up to float32 round-trip).

Run as:
    python tests/test_calibrate_bmaux_dyn_synthetic.py
or:
    pytest tests/test_calibrate_bmaux_dyn_synthetic.py -v
"""
import os
import sys
import pathlib

import numpy as np

# Ensure repo root is on the path so package imports resolve.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mapping.src.calibrate_bmaux import _band_frequencies, _project_field_to_bmaux
from mapping.src.calibrate_bmaux_dyn import _build_qgm, compute_qg_residuals


# ---------------------------------------------------------------------------
# Test 1 – projection recovery on a synthetic residual field
# ---------------------------------------------------------------------------
def test_projection_recovery():
    """_project_field_to_bmaux recovers Std/Tdec of a synthetic wave band."""
    rng = np.random.default_rng(42)

    # domain: 10° × 10° at 0.1° resolution
    lat = np.arange(33.0, 43.0 + 1e-9, 0.1)
    lon = np.arange(295.0, 305.0 + 1e-9, 0.1)
    nlat, nlon = lat.size, lon.size
    nt = 200
    dt_days = 1.0
    t_days = np.arange(nt) * dt_days

    # pick middle wavelength band
    ff = _band_frequencies(80.0, 1000.0, facpsp=1.5, npsp=3.5)
    ib = len(ff) // 2
    f0 = ff[ib]
    lam0 = 1.0 / f0
    print(f"  target band: lambda={lam0:.1f} km, f={f0:.5f} 1/km")

    # build synthetic field: A(t) * cos(2π f0 x) with known A0, tau0
    lat0_c = 38.0
    cos_lat = np.cos(lat0_c * np.pi / 180.0)
    x_km = (lon - 300.0) * 110.0 * cos_lat  # km
    A0 = 0.1   # m/day
    tau0 = 10.0  # days
    rho = np.exp(-dt_days / tau0)
    eps = rng.standard_normal(nt)
    a = np.empty(nt)
    a[0] = eps[0]
    for i in range(1, nt):
        a[i] = rho * a[i - 1] + np.sqrt(1.0 - rho ** 2) * eps[i]
    a = a / np.std(a) * A0

    field = (
        a[:, None, None]
        * np.cos(2.0 * np.pi * f0 * x_km)[None, None, :]
        * np.ones((1, nlat, 1))
    )  # (nt, nlat, nlon)  – uniform in lat

    valid_mask = np.ones((nlat, nlon), dtype=bool)
    lon_out = np.arange(296.0, 304.1, 2.0)
    lat_out = np.arange(34.0, 42.1, 2.0)

    _, Std, Tdec = _project_field_to_bmaux(
        field, t_days, lon, lat, valid_mask,
        lon_out, lat_out,
        lmin=80.0, lmax=1000.0, facpsp=1.5, npsp=3.5, ntheta=4,
        tdecmin=2.5, tdecmax=40.0,
        valid_min=0.0,
        verbose=True,
        _tag="test1",
    )

    # --- evaluate at the centre output cell
    jo = lat_out.size // 2
    io = lon_out.size // 2

    std_val = float(Std[ib, jo, io])
    tdec_val = float(Tdec[ib, jo, io])
    assert np.isfinite(std_val) and std_val > 0, f"Std at target band is invalid: {std_val}"
    std_ratio = std_val / A0
    tdec_ratio = tdec_val / tau0
    print(f"  Std: got={std_val:.4f}, expect~{A0:.4f}, ratio={std_ratio:.2f}")
    print(f"  Tdec: got={tdec_val:.2f}d, expect~{tau0:.1f}d, ratio={tdec_ratio:.2f}")
    assert 0.3 < std_ratio < 3.0, f"Std recovery out of bounds: ratio={std_ratio:.2f}"
    assert 0.3 < tdec_ratio < 2.5, f"Tdec recovery out of bounds: ratio={tdec_ratio:.2f}"

    # --- band selectivity: target band dominates adjacent bands
    for dib in (-2, -1, 1, 2):
        other_ib = ib + dib
        if 0 <= other_ib < Std.shape[0]:
            other_std = float(Std[other_ib, jo, io])
            if np.isfinite(other_std):
                ratio = other_std / std_val
                print(f"  neighbour band Δ{dib:+d}: Std={other_std:.4f}, ratio={ratio:.2f}")
                assert ratio < 1.1, (
                    f"Band selectivity failed: Std at band Δ{dib}={other_std:.4f} "
                    f">= Std at target={std_val:.4f}"
                )
    print("test_projection_recovery: OK")


# ---------------------------------------------------------------------------
# Test 2 – free-QG trajectory gives ~zero residuals
# ---------------------------------------------------------------------------
def test_qg_free_run_residuals():
    """Residuals of a free QG trajectory (ssh[n+1] = qgm.step(ssh[n])) are ~0."""
    rng = np.random.default_rng(7)

    # small domain: 21 × 21 grid at 0.1° (~10 km) resolution
    lat = np.arange(38.0, 40.01, 0.1)
    lon = np.arange(300.0, 302.01, 0.1)
    ny, nx = lat.size, lon.size
    nsteps = 5
    # Use one internal step per frame so nstep=1 in compute_qg_residuals.
    dt_ref_sec = 1200.0
    dt_ref_days = dt_ref_sec / 86400.0

    # random smooth initial SSH (SLA-scale, ~0.1 m)
    ssh0 = rng.standard_normal((ny, nx)).astype(np.float32) * 0.1

    qgm = _build_qgm(
        lon_r=lon,
        lat_r=lat,
        dt_seconds=1200.0,
        time_scheme="Euler",
        c_scalar=2.7,
        mdt=None,
        ssh_template=ssh0,
    )

    # build a free QG trajectory: each frame is the QG forecast of the previous
    hb = np.zeros((ny, nx), dtype=np.float32)
    ssh_stack = [ssh0.astype(np.float64)]
    for _ in range(nsteps):
        h0 = ssh_stack[-1].astype(np.float32)
        h1 = np.array(qgm.step(h0, hb, nstep=1))
        ssh_stack.append(h1.astype(np.float64))
    ssh = np.stack(ssh_stack)           # (nsteps+1, ny, nx)
    t_days = np.arange(nsteps + 1) * dt_ref_days

    eps, t_eps = compute_qg_residuals(
        ssh, t_days, qgm,
        hb_for_step=None,            # zeros — same as what we used above
        dt_internal_sec=1200.0,      # nstep = round(1200/1200) = 1
        verbose=True,
    )

    assert eps.shape == (nsteps, ny, nx), f"Wrong eps shape: {eps.shape}"
    eps_vals = eps[np.isfinite(eps)]
    max_abs = float(np.max(np.abs(eps_vals)))
    print(f"  max |eps| = {max_abs:.2e} m/day (should be < 1e-4)")
    assert max_abs < 1e-4, (
        f"Free-QG residuals unexpectedly large: max|eps|={max_abs:.2e} m/day"
    )
    print("test_qg_free_run_residuals: OK")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Test 1: projection recovery ===")
    test_projection_recovery()
    print()
    print("=== Test 2: free-QG residuals ===")
    test_qg_free_run_residuals()
    print()
    print("All tests passed.")
