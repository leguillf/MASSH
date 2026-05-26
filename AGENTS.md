# MASSH — Agent Guidelines

## Package overview

MASsh (Mapping SSH) is a Python framework for data assimilation of altimeter (and similar)
observations onto regular ocean grids, supporting BFN-QG, 4DVar, and forward-only experiments.

This document describes the package **excluding** the new outer-loop hyperparameter optimization
(`OUTER`, [outer.py](mapping/src/outer.py)).

## Package context

Before working on any task in this repository, load only the section(s) relevant to the task
from [doc/overview/](doc/overview/):

| Topic | File |
|-------|------|
| Architecture & component map | [doc/overview/01_high_level_design.md](doc/overview/01_high_level_design.md) |
| Config system, super-blocks, `config_default.py` | [doc/overview/02_configuration_system.md](doc/overview/02_configuration_system.md) |
| `State`, `Grid` | [doc/overview/03_state_and_grid.md](doc/overview/03_state_and_grid.md) |
| `MOD_QG1L_JAX`, `MOD_QGSW`, tracers, RK3, AD | [doc/overview/04_dynamical_models.md](doc/overview/04_dynamical_models.md) |
| Boundary conditions / `BC_EXT` | [doc/overview/05_boundary_conditions.md](doc/overview/05_boundary_conditions.md) |
| Observations, `OBSOP` | [doc/overview/06_observations.md](doc/overview/06_observations.md) |
| Reduced basis (`BASIS_*`) | [doc/overview/07_reduced_basis.md](doc/overview/07_reduced_basis.md) |
| 4DVar inversion, cost function, L-BFGS-B | [doc/overview/08_inversion.md](doc/overview/08_inversion.md) |
| Windowed tiled assimilation, multiprocessing | [doc/overview/09_run_time_orchestration.md](doc/overview/09_run_time_orchestration.md) |
| Diagnostics (`DIAG_OSSE`, `DIAG_OSE`) | [doc/overview/10_diagnostics.md](doc/overview/10_diagnostics.md) |
| Helper modules (`tools.py`, `switchvar.py`, …) | [doc/overview/11_auxiliary_helpers.md](doc/overview/11_auxiliary_helpers.md) |
| End-to-end usage example | [doc/overview/12_end_to_end_flow.md](doc/overview/12_end_to_end_flow.md) |
| Future work, tracer how-tos, adjoint tests | [doc/overview/13_notes_future_work.md](doc/overview/13_notes_future_work.md) |
| SLURM HPC execution (`slurm/`) | [doc/overview/14_slurm_hpc.md](doc/overview/14_slurm_hpc.md) |

## Source layout

```
mapping/src/          # Library (mod.py, inv.py, basis.py, …)
mapping/models/       # Dynamical model cores (sw.py, jqgm.py, …)
mapping/examples/     # Runnable experiments / config files
mapping/aux/          # Auxiliary fields (bathymetry, BM stats, …)
doc/overview/         # Per-section documentation (see table above)
slurm/                # SLURM HPC scripts for large-scale GPU-parallel runs
```

## Key conventions

- **JAX**: All `_JAX` model/basis/obsop variants must be used together on the 4DVar trajectory so autodiff traces end-to-end.
- **Area-scaling**: tracers and layer thicknesses are stored as `field * dx*dy` inside `sw.py`; convert back before I/O.
- **Shape conventions**: `State` uses `(ny, nx)` for h-grid fields; `sw.py` uses `(1, nl, nx, ny)` — convert with `.T` + `expand_dims`.
- **Adding a new variant**: add a `XXX_FOO = dict(super='XXX_FOO', …)` block to `mapping/src/config_default.py` and wire dispatch in the matching factory.
- **Syntax checks**: after editing `mod.py` or `sw.py` run `python -m py_compile <file>` before considering the task done.
