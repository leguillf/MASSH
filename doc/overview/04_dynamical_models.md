# 4. Dynamical models (`MOD`)

`Model(config, State)` ([mod.py](../../mapping/src/mod.py)) is a factory that returns
a model object exposing the standard interface:

- `Model.init(State, t=0)` — initialize prognostic variables
- `Model.step(State, nstep, t=...)` — time-march `nstep` steps of size `Model.dt`
- `Model.set_bc(t_bc, var_bc)` — inject lateral/large-scale BCs
- `Model.save_output(...)` — write a snapshot to NetCDF
- `Model.dt`, `Model.timestamps`, `Model.T`, `Model.var_to_save`

JAX models additionally provide `jit`-compiled tangent-linear / adjoint
operators (built from `jax.jvp` / `jax.vjp`), which are what `INV_4DVAR` uses.

Concrete implementations live in [mapping/models/](../../mapping/models):
- `model_qg1l/` — single-layer quasi-geostrophic
- `model_qgsw/` — shallow-water (SW) model; also supports QG-class runs
- `model_sw1l/` — single-layer shallow water (with internal-tide variants)

---

## 4a. `MOD_QG1L_JAX` — quasi-geostrophic model with optional tracers

Implemented in `Model_qg1l_jax` ([mod.py](../../mapping/src/mod.py)).
Propagates SSH via `Qgm_trac` ([jqgm.py](../../mapping/models/model_qg1l/jqgm.py))
and optionally one or more passive tracers (e.g. SST, SSS) by advection with
the geostrophic velocity field.

Key config parameters (`MOD_QG1L_JAX` block):

| Parameter | Default | Description |
|---|---|---|
| `name_var` | `{'SSH': 'ssh'}` | Variable mapping; add tracers as extra keys |
| `ageo_velocities` | `False` | Include ageostrophic velocities in the tracer advection |
| `forcing_tracer_from_bc` | `False` | Add a nudging term `Fc * (Xb − X)` from BCs |
| `sponge_coef` | `0.` | Rayleigh-damping rate at the sponge boundary (units: per step) |

Boundary treatment: `Wbc` (a `[0,1]` weight map from `grid.compute_weight_map`)
controls where sponge nudging is applied. When `sponge_coef > 0` the sponge
nudging `sponge_coef * Wbc * (Xb − X)` is applied before the forcing flux
`(1 − Wbc) * Fc` — consistently in both `step()` and `step_jax()`.

---

## 4b. `MOD_QGSW` — shallow-water model with passive tracer advection

Implemented in `Model_qgsw` ([mod.py](../../mapping/src/mod.py)), wrapping the
JAX shallow-water core [sw.py](../../mapping/models/model_qgsw/sw.py).

**Dynamics**: WENO-6 advection, SSP-RK3 time integration, `lax.scan` +
`jax.checkpoint` memory-efficient time loop. Supports multi-layer (`nl > 1`)
and single-layer configurations. TGL/ADJ are obtained automatically via
`jax.jvp` / `jax.vjp` on the JIT-compiled forward core.

**Passive tracer advection** (new): any `name_var` entries that are not `U`,
`V`, or `SSH` are treated as passive tracers. When at least one is present,
`Model_qgsw` sets `advect_tracer=True` and activates the tracer path:

- Tracer variables hold physical values on the `(ny, nx)` h-grid.
- `sw.py` stores tracers area-scaled internally (`c_area = c_phys * dx*dy`),
  consistent with the h convention, to reuse the WENO flux machinery.
- Advection uses the **surface physical velocity** (`U_h`, `V_h`) derived from
  the staggered `(u, v)` fields via `compute_diagnostic_variables` — the same
  call already needed for the momentum RHS, so there is no extra cost per stage.
- Diffusion applies a Laplacian `κ∇²c_phys` via `add_tracer_diffusion`.
- Sponge Rayleigh damping `γ(c_b − c)` is applied at each RK3 sub-stage,
  using the same `sponge_h` mask as the height field.
- The joint `(u, v, h, c)` integration is `sw.step_with_tracer`; the
  AD-compatible wrappers `jstep_core_trac`, `jstep_tgl_trac`,
  `jstep_adj_trac` mirror `jstep_core` / `jstep_tgl` / `jstep_adj`.

Key config parameters added to `MOD_QGSW`:

| Parameter | Default | Description |
|---|---|---|
| `diff_coef_trac` | `0.` | Laplacian diffusivity for tracers (m² s⁻¹) |

(The existing `sponge_coef` and `dist_sponge_bc` already apply to tracers.)

### Shape and coordinate conventions

| Layer | Shape (h-grid) | Shape (u-grid) | Shape (v-grid) |
|---|---|---|---|
| State (`mod.py`) | `(ny, nx)` | `(ny, nx+1)` | `(ny+1, nx)` |
| SW internal (`sw.py`) | `(1, nl, nx, ny)` | `(1, nl, nx+1, ny)` | `(1, nl, nx, ny+1)` |

Conversion: `.T` + `expand_dims`. Tracers follow the h-grid in both layers.
Area-scaling: `c_area = c_phys * dx*dy`, same convention as `h = h_phys * area`.

### SSP-RK3 delta form in `step_with_tracer`

Stage 0 (Euler predictor):
```
u1 = u0 + dt * dt0
```
Stage 1:
```
u2 = u1 + (dt/4) * (dt1 - 3*dt0)
```
Stage 2 (final):
```
u_new = u2 + (dt/12) * (8*dt2 - dt1 - dt0)
```

The `_stage_tendencies(u, v, h, c)` helper calls `compute_diagnostic_variables`
once per stage and reuses `U, V` for both momentum and tracer advection (no
extra diagnostic call per stage).
