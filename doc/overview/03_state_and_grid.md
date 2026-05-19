# 3. State and Grid

`State` ([state.py](../../mapping/src/state.py)) wraps the grid and a `params` dict
holding prognostic variables (typically `SSH`, possibly `u`, `v`, layer
thicknesses, etc.). It exposes `.copy()`, `.plot()`, and serialization helpers.

`grid` ([grid.py](../../mapping/src/grid.py)) provides:
- `lonlat2dxdy`, `dxdy2xy` — metric conversions
- KDTree-based nearest-neighbor utilities for irregular obs interpolation
- Geographical (`GRID_GEO`) and Cartesian (`GRID_CAR`) grid construction.
