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
