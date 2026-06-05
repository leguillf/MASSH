# jqgm.py — Qgm formulation refactor (VarDyn, May 2026)

## What changed
- `Qgm_sf` class deleted. `Qgm` now has a `formulation` parameter (`'ssh'` | `'sf'`).
- `config_default.py` → `MOD_QG1L_JAX` gained `formulation = 'ssh'`.
- `mod.py` → constructor call passes `formulation=config.MOD.formulation`.

## Formulation-aware methods in `Qgm`
| Method | `'ssh'` | `'sf'` |
|---|---|---|
| `helmoltz_dst` | `g/f0 · L − g·f0/c²` | `L − (f0/c)²` |
| `h2uv` | `g/f0` scalar | compute `φ=g/f·h`, then differentiate `φ` |
| `h2pv` | `g/f0 · ∇²h − (f0/c)²·h` | `∇²φ − (f0/c)²·φ`, φ=g/f·h |
| `pv2h` | direct h via dst | invert to φ, then h=(f/g)·φ |
| `rhs` bathymetry | `f0 · bathymetry_PV_term` | `f · bathymetry_PV_term` |

## Bug fixed (commit baec2d6)
`h2uv` was missing the formulation branch — used full 2-D `self.f` (ny,nx)
against stencil slices (ny-2, nx-1) → `TypeError` with `formulation='sf'`.
Initial fix used sliced `self.f`; follow-up audit restored old `Qgm_sf` semantics:
`'sf'` differentiates `φ=g/f·h`, while `'ssh'` uses scalar `self.f0`.

## Commits
- `40d3066` — refactor: merge Qgm_sf into Qgm via formulation='sf' parameter
- `baec2d6` — fix: add formulation branch to Qgm.h2uv to avoid shape mismatch with 2D f

## Smoke test (future prompts)
Use this when validating `mapping/models/model_qg1l/jqgm.py` changes.

1. Activate env and set import path
```bash
source ~/.bashrc >/dev/null 2>&1 || true
conda activate massh
cd /home/flo/MASSH
export PYTHONPATH=/home/flo/MASSH/mapping
```

2. Run the smoke script
```bash
python - <<'PY'
import numpy as np
import jax.numpy as jnp
from mapping.models.model_qg1l.jqgm import Qgm

ny, nx = 10, 12
dx = jnp.ones((ny, nx))*10000.
dy = jnp.ones((ny, nx))*10000.
f = jnp.ones((ny, nx))*1e-4
c = jnp.ones((ny, nx))*2.7
ssh = jnp.zeros((ny, nx))

m = Qgm(dx=dx, dy=dy, dt=600., SSH=np.array(ssh), c=np.array(c), f=np.array(f),
	formulation='ssh', advect_pv=True, ageo_velocities=False, time_scheme='rk3')
x0 = jnp.zeros((1, ny, nx))
xb = jnp.zeros((1, ny, nx))
out = m.step(x0, xb, nstep=2)
print('stacked_shape', out.shape)

m2 = Qgm(dx=dx, dy=dy, dt=600., SSH=np.array(ssh), c=np.array(c), f=np.array(f),
	 formulation='sf', advect_pv=True, ageo_velocities=False, time_scheme='rk2')
out2 = m2.step(jnp.zeros((ny,nx)), jnp.zeros((ny,nx)), nstep=2)
print('ssh_shape', out2.shape)
PY
```

Expected output:
- `stacked_shape (1, 10, 12)`
- `ssh_shape (10, 12)`

Notes:
- `PYTHONPATH=/home/flo/MASSH/mapping` is required so `from src.config import USE_FLOAT64` resolves.
- `python -m py_compile mapping/models/model_qg1l/jqgm.py` is a quick syntax check before running the smoke test.

## Tracer advection physical consistency in QG

- The tracer equation in `Qgm.rhs` is advective form (`dc/dt = -u·∇c`) using the
	same upwind operator as PV.
- For geostrophic velocities used by QG, flow is non-divergent in both
	formulations:
	- `'ssh'`: `u = -g/f0 * dh/dy`, `v = g/f0 * dh/dx` (scalar `f0`)
	- `'sf'`: `u = -dphi/dy`, `v = dphi/dx`
	so advective and conservative forms are equivalent at the continuous level.
- Unlike SW, QG tracer transport does not need an extra `c*div(u)` correction
	because `div(u) ≈ 0` for geostrophic flow.
- Land/coast consistency is enforced by three mechanisms:
	velocity masking on coastal T-points (`ocean_h`), land clamp in `bc()`
	(`var1[ind0] = varb[ind0]`), and nearest-neighbour BC fill in
	`Model_qg1l_jax.set_bc`.

## BC stack convention (Model_qg1l_jax -> Qgm)

- `Qgm.step` expects boundary stack layout:
	`Xb = [SSH_bc, tracer_bc...]`.
- Ageostrophic `U/V` are carried in `X0` state and are not part of `Xb`.
- `_apply_bc_jax` was aligned to this convention to match `_apply_bc` and avoid
	ageostrophic+tracer indexing mismatches.
