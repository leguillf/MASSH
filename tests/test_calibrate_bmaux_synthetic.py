"""Synthetic test for calibrate_bmaux.

Builds a single-wavelet SSH field with known amplitude A0 and decorrelation
time tau0, runs the calibrator on the band it lives in, and checks Std/Tdec.
"""
import os, tempfile
import numpy as np
import xarray as xr

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mapping.src.calibrate_bmaux import calibrate_bmaux, _band_frequencies


def main():
    rng = np.random.default_rng(0)
    # --- domain
    lat = np.arange(33.0, 43.0 + 1e-9, 0.1)
    lon = np.arange(295.0, 305.0 + 1e-9, 0.1)
    nlat, nlon = lat.size, lon.size
    nt = 200
    dt_days = 1.0
    t_days = np.arange(nt) * dt_days
    time = np.array(t_days * 86400.0, dtype="timedelta64[s]") + np.datetime64("2012-10-01")

    # pick band
    ff = _band_frequencies(80.0, 1000.0, facpsp=1.5, npsp=3.5)
    ib = len(ff) // 2
    f0 = ff[ib]            # 1/km
    lam0 = 1.0 / f0
    print(f"target band: lambda={lam0:.1f} km, f={f0:.5f} 1/km")

    # build a plane wave SSH = A(t)*cos(2*pi*f0 * x)  (theta=0)
    lat0_c = 38.0
    cos_lat = np.cos(lat0_c * np.pi / 180.0)
    x_km = (lon - 300.0) * 110.0 * cos_lat  # km, centred near domain mid
    A0 = 0.2
    tau0 = 10.0  # days
    # AR(1)-like envelope with target 1/e decorrelation = tau0
    rho = np.exp(-dt_days / tau0)
    eps = rng.standard_normal(nt)
    a = np.empty(nt)
    a[0] = eps[0]
    for i in range(1, nt):
        a[i] = rho * a[i - 1] + np.sqrt(1 - rho * rho) * eps[i]
    a = a / np.std(a) * A0  # rescale to unit std A0

    ssh = (a[:, None, None] * np.cos(2 * np.pi * f0 * x_km)[None, None, :]
           * np.ones((1, nlat, 1)))

    ds = xr.Dataset(
        {"ssh": (("time", "lat", "lon"), ssh.astype("float32"))},
        coords={"time": time, "lat": lat.astype("float32"), "lon": lon.astype("float32")},
    )
    with tempfile.TemporaryDirectory() as td:
        ref = os.path.join(td, "ref.nc")
        out = os.path.join(td, "aux.nc")
        ds.to_netcdf(ref)
        res = calibrate_bmaux(
            ref_path=ref, out_path=out,
            lon_min=297.0, lon_max=303.0, lat_min=35.0, lat_max=41.0,
            lmin=80.0, lmax=1000.0, facpsp=1.5, npsp=3.5,
            ntheta=4, dlon_out=2.0, dlat_out=2.0,
            tdecmin=0.1, tdecmax=200.0, lon_unit_out="0_360",
            verbose=False,
        )
        std_band = res["Std"].isel(f=ib).values
        tdec_band = res["Tdec"].isel(f=ib).values
        # Expected Std: only theta=0 carries variance ≈ A0^2; averaged over
        # ntheta and divided by 2 (Cc+Cs combined) -> Std ≈ A0/sqrt(2*ntheta)
        ntheta = 4
        std_expect = A0 / np.sqrt(2 * ntheta)
        print(f"Std at target band : mean={np.nanmean(std_band):.4f}  (expect ~{std_expect:.4f})")
        print(f"Tdec at target band: mean={np.nanmean(tdec_band):.2f}d (expect ~{tau0:.1f} d)")
        for db in (-2, -1, 1, 2):
            ib2 = ib + db
            if 0 <= ib2 < ff.size:
                s2 = np.nanmean(res["Std"].isel(f=ib2).values)
                print(f"  neighbour band {db:+d} (lambda={1/ff[ib2]:.0f}km): Std={s2:.4f}")
        std_ratio = np.nanmean(std_band) / std_expect
        tdec_ratio = np.nanmean(tdec_band) / tau0
        # band-selectivity: target band should dominate neighbours
        neighbour_std = max(
            np.nanmean(res["Std"].isel(f=ib + db).values)
            for db in (-2, -1, 1, 2)
            if 0 <= ib + db < ff.size
        )
        assert np.nanmean(std_band) > neighbour_std, (
            f"target band Std {np.nanmean(std_band):.4f} not dominant "
            f"over neighbours ({neighbour_std:.4f})"
        )
        assert 0.4 < std_ratio < 2.5, f"Std off: ratio={std_ratio:.2f}"
        assert 0.4 < tdec_ratio < 2.0, f"Tdec off: ratio={tdec_ratio:.2f}"
        print("OK")


if __name__ == "__main__":
    main()
