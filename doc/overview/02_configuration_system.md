# 2. Configuration system

[exp.py](../../mapping/src/exp.py) defines a small `Config(dict)` class whose keys are
exposed as attributes. A user config file declares e.g.

```python
NAME_GRID  = 'GRID_GEO'
NAME_MOD   = 'MOD_QG1L_JAX'
NAME_OBS   = ['SWOT', 'NADIRS']
NAME_OBSOP = 'OBSOP_INTERP_L3_JAX'
NAME_BASIS = 'BASIS_BMaux_JAX'
NAME_INV   = 'INV_4DVAR'

GRID_GEO = dict(super='GRID_GEO', lon_min=..., dlon=..., ...)
SWOT     = dict(super='OBS_SSH_SWATH', path=..., name_var=..., ...)
...
```

`Exp(path_config)` loads the user file and `merge_configs(...)` (in
[exp.py](../../mapping/src/exp.py)) recursively merges it with the defaults from
[config_default.py](../../mapping/src/config_default.py). For block names that are
*lists* (like `NAME_OBS`), each entry is merged independently — this is how
multiple obs sources coexist. Nested dicts containing a `super` field are also
merged recursively.

Selecting a `super=` value picks an implementation; defaults come from the
matching dictionary in `config_default.py`.

## Registered super-blocks (current state)

- **GRID**: `GRID_FROM_FILE`, `GRID_GEO`, `GRID_CAR`, `GRID_CAR_CENTER`, `GRID_RESTART`
- **OBS**: `OBS_L4`, `OBS_SSH_NADIR`, `OBS_SSH_SWATH`
- **MOD**: `MOD_Id`, `MOD_DIFF`, `MOD_DIFF_JAX`, `MOD_QG1L_JAX`, `MOD_CSW1L`,
  `MOD_QGSW`, `MOD_BMIT`
- **BC**: `BC_EXT`
- **OBSOP**: `OBSOP_INTERP_L3`, `OBSOP_INTERP_L3_JAX`, `OBSOP_INTERP_L4`
- **INV**: `INV_4DVAR`
- **BASIS**: `BASIS_BM`, `BASIS_BM_JAX`, `BASIS_GAUSSV2`, `BASIS_GAUSS3D`,
  `BASIS_GAUSS3D_JAX`, `BASIS_MIOST`, `BASIS_MIOST_JAX`, `BASIS_WAVELET3D`,
  `BASIS_BMaux`, `BASIS_BMaux_JAX`, `BASIS_HBC_JAX`, `BASIS_HBC_CST_JAX`,
  `BASIS_OFFSET`, `BASIS_OFFSET_JAX`
- **DIAG**: `DIAG_OSSE`, `DIAG_OSE`

The `_JAX` variants are JAX-traceable so they can be used inside the 4DVar
adjoint computation (autodiff).

A global float precision flag lives in [config.py](../../mapping/src/config.py)
(`USE_FLOAT64`), which propagates to JAX.
