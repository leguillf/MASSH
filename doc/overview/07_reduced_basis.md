# 7. Reduced basis (`BASIS`)

The control vector of the inversion is *not* the full model state — it is the
coefficients of a reduced basis defined in [basis.py](../../mapping/src/basis.py).
Each basis super-block exposes the same interface:

- `Basis.set_basis(...)` — build basis structure (locations, wavelet shapes,
  per-coefficient priors). For `BASIS_BMaux*`, this also caches background
  hyperparameters from `aux_reduced_basis_BM.nc` (`tdec`, `std`).
- `Basis.operg(t_days, X, State=...)` — apply the basis (control → grid) at
  time `t`, used at every checkpoint of the model integration.
- Attributes: `Basis.NP` (number of components per frequency), `Basis.nf`
  (number of frequencies), `Basis.Q` (prior covariance diagonal), and (for
  BMaux variants) the cached `tdec`, `std`, `Q_bg` arrays.

Common families:
- `BM` / `BMaux` — wavelet/coarse-graining basis driven by background SSH
  statistics; `BMaux*` reads pre-computed auxiliary statistics.
- `GAUSS*` — radial-basis Gaussians on a reduced grid.
- `MIOST` — multiscale optimally interpolated stationary basis.
- `WAVELET3D` — 3D (space–time) wavelet basis.
- `OFFSET`, `HBC` — offsets / BC-correction bases used in compound setups.

Multiple basis blocks can be combined when `NAME_BASIS` is a list (the
framework wraps them in a multi-mode container).
