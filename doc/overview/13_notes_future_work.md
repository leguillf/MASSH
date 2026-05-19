# 13. Notes for future work

- Every pipeline component is selected by a `super` string and merged with
  defaults; **adding a new model/basis/obsop variant** boils down to (i)
  adding a `XXX_FOO = dict(super='XXX_FOO', ...)` block to
  [config_default.py](../../mapping/src/config_default.py) and (ii) wiring the
  dispatch in the matching `Module(config, ...)` factory.
- `_JAX` variants are required everywhere on the trajectory if you want the
  4DVar adjoint to be autodiffed end-to-end — `Inv_4Dvar` traces through
  `Basis.operg`, `Model.step`, and `Obsop.misfit`.
- Cached files (`Obsop` H-matrices, `Basis` aux files) live under
  `EXP.tmp_DA_path`; their cache keys depend on the joined `name_obs` list,
  the basis hyperparameters, and the grid — invalidate by changing any of
  these or wiping the directory.
- The control vector is persisted as `Xres.nc` under
  `INV.path_save_control_vectors` (or `EXP.tmp_DA_path`); this file is the
  natural hand-off point for re-evaluation, warm starts, and outer-loop work.

---

## Adding passive tracers to `MOD_QGSW`

Declare the tracer alongside `SSH`/`U`/`V` in `name_var`, provide matching
entries in `name_var_bc`, and (optionally) set `diff_coef_trac` in the config:

```python
MOD_QGSW = dict(
    super = 'MOD_QGSW',
    name_var = {'SSH': 'ssh', 'U': 'u', 'V': 'v', 'SST': 'sst'},
    name_var_bc = {'SSH': 'ssh', 'U': 'u', 'V': 'v', 'SST': 'sst'},
    diff_coef_trac = 100.,   # m² s⁻¹
    ...
)
```

Any key in `name_var` that is not `U`, `V`, or `SSH` is automatically routed
through the tracer advection path (`advect_tracer=True`). Boundary conditions
for tracers are set via the same `Bc` / `set_bc` pipeline as SSH. TGL and ADJ
propagate through the tracer fields automatically via `jstep_tgl_trac` /
`jstep_adj_trac` (JAX jvp/vjp on `jstep_core_trac`), so tracers are fully
4DVar-compatible.

---

## Adding passive tracers to `MOD_QG1L_JAX`

Similarly extend `name_var` with the tracer key(s). Additional options:

```python
MOD_QG1L_JAX = dict(
    super = 'MOD_QG1L_JAX',
    name_var = {'SSH': 'ssh', 'SST': 'sst'},
    ageo_velocities = False,        # include ageostrophic velocities in tracer step
    forcing_tracer_from_bc = False, # nudge tracers toward BCs: Fc*(Xb-X)
    sponge_coef = 0.,               # Rayleigh damping weight at boundaries
    ...
)
```

`sponge_coef * Wbc * (Xb − X)` is applied to *all* prognostic variables
(SSH + tracers) before the basis forcing flux, in both `step()` and `step_jax()`.

---

## Potential next steps

- Runtime testing: instantiate `Model_qgsw` with a tracer variable and run a
  short forward integration to validate correctness.
- Adjoint test for tracer path: extend `adjoint_test_jstep` in
  [mod.py](../../mapping/src/mod.py) to cover `jstep_tgl_trac` / `jstep_adj_trac`.
- Integration with `Basis.operg` for tracer control variables (currently
  tracers receive forcing from `State.params` but are not yet part of a
  reduced basis).
- Config example file showing tracer advection setup end-to-end.
