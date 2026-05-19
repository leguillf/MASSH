# 5. Boundary conditions (`BC`)

[bc.py](../../mapping/src/bc.py) — `Bc(config, State)` returns a BC object whose
`interp(times)` method produces the large-scale field at the requested
timestamps. The current implementation `BC_EXT` reads an external NetCDF (e.g.
DUACS L4) and provides spatial+temporal interpolation. The model's `set_bc`
absorbs this into the lateral relaxation / nudging machinery.
