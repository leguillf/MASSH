# 1. High-level design

MASSH is a configuration-driven assimilation framework. A run is fully specified
by a Python configuration file that sets a handful of *block names* (`NAME_*`),
each of which selects an implementation among a registry of *super-blocks*
defined in [config_default.py](../../mapping/src/config_default.py).

The framework wires together the following modular components:

| Component  | Module                                | Role                                                    |
|------------|---------------------------------------|---------------------------------------------------------|
| `EXP`      | [exp.py](../../mapping/src/exp.py)          | Experiment metadata (paths, dates, save options)        |
| `GRID`     | [grid.py](../../mapping/src/grid.py)        | Spatial grid (Cartesian / geographical / from file)     |
| `State`    | [state.py](../../mapping/src/state.py)      | Holds prognostic variables on the grid                  |
| `MOD`      | [mod.py](../../mapping/src/mod.py)          | Forward dynamical model (QG, SW, diffusion, identity)   |
| `BC`       | [bc.py](../../mapping/src/bc.py)            | Boundary / large-scale conditions                       |
| `OBS`      | [obs.py](../../mapping/src/obs.py)          | Reads & preprocesses observations into a `dict_obs`     |
| `OBSOP`    | [obsop.py](../../mapping/src/obsop.py)      | Observation operator H (model → obs space)              |
| `BASIS`    | [basis.py](../../mapping/src/basis.py)      | Reduced control basis (wavelets, Gaussians, MIOST, ...) |
| `INV`      | [inv.py](../../mapping/src/inv.py)          | Inversion algorithm (4DVar, BFN, forward only)          |
| `DIAG`     | [diag.py](../../mapping/src/diag.py)        | Post-run diagnostics (OSSE / OSE)                       |

A high-level orchestrator [run_assimilation.py](../../mapping/src/run_assimilation.py)
provides parallel windowed assimilation (overlapping space–time tiles, merging,
multiprocessing).

```
config.py ──► exp.Exp() ──► merge with config_default
                            │
                            ▼
   ┌──── State ◄── GRID ────┴──── EXP
   │
   ├──── Bc       ── BC
   ├──── Model    ── MOD
   ├──── dict_obs ── OBS
   ├──── Obsop    ── OBSOP   (uses dict_obs, State, Model)
   ├──── Basis    ── BASIS   (reduced control space)
   │
   └──► Inv(...)  ── INV     (4DVar, BFN, forward)
                            │
                            ▼
                          Diag
```
