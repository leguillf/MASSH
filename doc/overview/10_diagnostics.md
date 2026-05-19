# 10. Diagnostics (`DIAG`)

[diag.py](../../mapping/src/diag.py) — `Diag(config, State)` post-processes outputs:

- `DIAG_OSSE` — twin-experiment metrics against a known truth (RMSE, spectra,
  effective resolution `λx`, time-mean / space-mean error maps, animations).
- `DIAG_OSE` — observation-space metrics for real-data runs (independent
  altimeter cross-validation, along-track power spectra).

Plots use cartopy + cmocean; spectra rely on `xrft`. Heavy diagnostics are
parallelized through `joblib.Parallel`.
