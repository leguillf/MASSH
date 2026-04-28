# MASSH — Package Overview

> Mapping SSH (Sea Surface Height) — a Python framework for data assimilation of
> altimeter (and similar) observations onto regular ocean grids, supporting
> BFN-QG, 4DVar, and forward-only experiments.
>
> Source layout: [mapping/src/](mapping/src) (library), [mapping/models/](mapping/models)
> (dynamical model implementations), [mapping/examples/](mapping/examples) (runnable
> experiments), [mapping/aux/](mapping/aux) (auxiliary fields).

This document describes the package as it stands on the current branch
**excluding** the new outer-loop hyperparameter optimization
(`OUTER`, [outer.py](mapping/src/outer.py)). It is intended as a reference for
future work building on the existing pipeline.

---

## 1. High-level design

MASSH is a configuration-driven assimilation framework. A run is fully specified
by a Python configuration file that sets a handful of *block names* (`NAME_*`),
each of which selects an implementation among a registry of *super-blocks*
defined in [config_default.py](mapping/src/config_default.py).

The framework wires together the following modular components:

| Component  | Module                                | Role                                                    |
|------------|---------------------------------------|---------------------------------------------------------|
| `EXP`      | [exp.py](mapping/src/exp.py)          | Experiment metadata (paths, dates, save options)        |
| `GRID`     | [grid.py](mapping/src/grid.py)        | Spatial grid (Cartesian / geographical / from file)     |
| `State`    | [state.py](mapping/src/state.py)      | Holds prognostic variables on the grid                  |
| `MOD`      | [mod.py](mapping/src/mod.py)          | Forward dynamical model (QG, SW, diffusion, identity)   |
| `BC`       | [bc.py](mapping/src/bc.py)            | Boundary / large-scale conditions                       |
| `OBS`      | [obs.py](mapping/src/obs.py)          | Reads & preprocesses observations into a `dict_obs`     |
| `OBSOP`    | [obsop.py](mapping/src/obsop.py)      | Observation operator H (model → obs space)              |
| `BASIS`    | [basis.py](mapping/src/basis.py)      | Reduced control basis (wavelets, Gaussians, MIOST, ...) |
| `INV`      | [inv.py](mapping/src/inv.py)          | Inversion algorithm (4DVar, BFN, forward only)          |
| `DIAG`     | [diag.py](mapping/src/diag.py)        | Post-run diagnostics (OSSE / OSE)                       |

A high-level orchestrator [run_assimilation.py](mapping/src/run_assimilation.py)
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

---

## 2. Configuration system

[exp.py](mapping/src/exp.py) defines a small `Config(dict)` class whose keys are
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
[exp.py](mapping/src/exp.py)) recursively merges it with the defaults from
[config_default.py](mapping/src/config_default.py). For block names that are
*lists* (like `NAME_OBS`), each entry is merged independently — this is how
multiple obs sources coexist. Nested dicts containing a `super` field are also
merged recursively.

Selecting a `super=` value picks an implementation; defaults come from the
matching dictionary in `config_default.py`.

### Registered super-blocks (current state)

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

A global float precision flag lives in [config.py](mapping/src/config.py)
(`USE_FLOAT64`), which propagates to JAX.

---

## 3. State and Grid

`State` ([state.py](mapping/src/state.py)) wraps the grid and a `params` dict
holding prognostic variables (typically `SSH`, possibly `u`, `v`, layer
thicknesses, etc.). It exposes `.copy()`, `.plot()`, and serialization helpers.

`grid` ([grid.py](mapping/src/grid.py)) provides:
- `lonlat2dxdy`, `dxdy2xy` — metric conversions
- KDTree-based nearest-neighbor utilities for irregular obs interpolation
- Geographical (`GRID_GEO`) and Cartesian (`GRID_CAR`) grid construction.

---

## 4. Dynamical models (`MOD`)

`Model(config, State)` ([mod.py](mapping/src/mod.py)) is a factory that returns
a model object exposing the standard interface:

- `Model.init(State, t=0)` — initialize prognostic variables
- `Model.step(State, nstep, t=...)` — time-march `nstep` steps of size `Model.dt`
- `Model.set_bc(t_bc, var_bc)` — inject lateral/large-scale BCs
- `Model.save_output(...)` — write a snapshot to NetCDF
- `Model.dt`, `Model.timestamps`, `Model.T`, `Model.var_to_save`

JAX models additionally provide `jit`-compiled tangent-linear / adjoint
operators (built from `jax.jvp` / `jax.vjp`), which are what `INV_4DVAR` uses.

Concrete implementations live in [mapping/models/](mapping/models):
- `model_qg1l/` — single-layer quasi-geostrophic
- `model_qgsw/` — QG + shallow-water coupled
- `model_sw1l/` — single-layer shallow water (with internal-tide variants)

---

## 5. Boundary conditions (`BC`)

[bc.py](mapping/src/bc.py) — `Bc(config, State)` returns a BC object whose
`interp(times)` method produces the large-scale field at the requested
timestamps. The current implementation `BC_EXT` reads an external NetCDF (e.g.
DUACS L4) and provides spatial+temporal interpolation. The model's `set_bc`
absorbs this into the lateral relaxation / nudging machinery.

---

## 6. Observations (`OBS`) and observation operator (`OBSOP`)

`Obs(config, State)` ([obs.py](mapping/src/obs.py)) returns `dict_obs`, a
dictionary keyed by observation timestamp; values are dicts holding the file
paths, source name, variable name, and metadata required to assimilate that
batch. Three super-blocks exist:

- `OBS_SSH_NADIR` — 1D nadir-altimeter tracks
- `OBS_SSH_SWATH` — 2D SWOT-like swaths
- `OBS_L4` — gridded L4 (used as ground truth in OSSE diagnostics or as a
  validation reference)

`Obsop(config, State, dict_obs, Model)` ([obsop.py](mapping/src/obsop.py))
builds the observation operator H. It precomputes per-timestamp interpolation
weights and caches them on disk (the cache key contains the joined `name_obs`
list, so disjoint subsets do not collide). Public methods used elsewhere:

- `Obsop.process_obs()` — finalize cache
- `Obsop.is_obs_time(t)` — fast membership test
- `Obsop.misfit(t, State)` — `(y - H x) / σ` evaluated at time `t`

`OBSOP_INTERP_L3*` interpolates the model state onto sparse along-track /
swath observations; `OBSOP_INTERP_L4` interpolates onto a regular L4 grid.

---

## 7. Reduced basis (`BASIS`)

The control vector of the inversion is *not* the full model state — it is the
coefficients of a reduced basis defined in [basis.py](mapping/src/basis.py).
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

---

## 8. Inversion (`INV`)

[inv.py](mapping/src/inv.py) defines:

- `Inv_forward(config, State, Model, Bc)` — pure forward integration with
  output saving (used when `NAME_INV` is `None`).
- `Inv_4Dvar(config, State, Model, dict_obs, Obsop, Basis, Bc)` —
  incremental 4DVar in the reduced basis.

### 4DVar in MASSH

The control vector `X` lives in basis space. The cost function is

$$ J(X) = \tfrac12 X^\top Q^{-1} X + \tfrac12 \sum_t \big\| (y_t - H_t M_{0\to t}(\Phi(X))) / \sigma_t \big\|^2 $$

where $\Phi$ = `Basis.operg`, $M$ = `Model.step`, $H$ = `Obsop`, and $Q$ is the
diagonal prior carried by the basis. Gradients are obtained by combining the
basis adjoint, JAX `vjp` of the JAX-models, and the obsop adjoint. The outer
optimizer is `scipy.optimize.minimize` with method `L-BFGS-B`, controlled by:

- `INV.maxiter` — outer iteration cap
- `INV.ftol`, `INV.gtol` — stopping tolerances
- `INV.timestep_checkpoint` — frequency at which `Basis.operg` is reapplied
  along the trajectory
- `INV.path_init_4Dvar` — optional warm start (`Xres.nc` from a prior run)
- `INV.path_save_control_vectors` — destination for `Xres.nc` (final control
  vector, used by downstream diagnostics and reruns).

The custom exceptions `ConvergenceReached` / `CrazyGradient` short-circuit the
optimizer when `ftol`/`gtol` are met or when a NaN/Inf gradient is detected.

---

## 9. Run-time orchestration

[run_assimilation.py](mapping/src/run_assimilation.py) implements **windowed
assimilation**: it splits the full domain into overlapping space–time tiles
(`time_window_size_proc`, `space_window_size_proc_*`, `*_overlap`), runs each
tile as an independent `Inv_4Dvar` job (optionally on a chosen GPU), and
merges the outputs by Gaspari–Cohn-tapered weighted averaging. Equatorial
tiles can use an alternate config (`config_eq`) so a different model can be
applied across the equator.

Key features:
- `flag_init_from_previous` — chain time windows so each starts from the
  previous one's final state.
- `flag_init`, `flag_background` — seed from another experiment.
- Multiprocessing (`nx_proc`, `ny_proc`, `gpu_devices`).
- Pickle-based job spec dumped to `dir_save_pickle` for restart.

---

## 10. Diagnostics (`DIAG`)

[diag.py](mapping/src/diag.py) — `Diag(config, State)` post-processes outputs:

- `DIAG_OSSE` — twin-experiment metrics against a known truth (RMSE, spectra,
  effective resolution `λx`, time-mean / space-mean error maps, animations).
- `DIAG_OSE` — observation-space metrics for real-data runs (independent
  altimeter cross-validation, along-track power spectra).

Plots use cartopy + cmocean; spectra rely on `xrft`. Heavy diagnostics are
parallelized through `joblib.Parallel`.

---

## 11. Auxiliary helpers

- [tools.py](mapping/src/tools.py) — Gaspari–Cohn tapers, detrending,
  auxiliary-data readers.
- [tools_4Dvar.py](mapping/src/tools_4Dvar.py) — gradient-test scaffolding,
  J/grad-J wrappers around the basis+model+obsop chain.
- [tools_bfn.py](mapping/src/tools_bfn.py) — back-and-forth nudging utilities
  (legacy BFN-QG path).
- [switchvar.py](mapping/src/switchvar.py) — variable transforms (SSH ↔ PV ↔
  velocity) used by some diagnostics and bases.

---

## 12. Typical end-to-end flow

```python
from src import exp, state, mod, bc, obs, obsop, basis, inv, diag

config   = exp.Exp('config_my_run.py')
State    = state.State(config)
Model    = mod.Model(config, State)
Bc       = bc.Bc(config, State)
dict_obs = obs.Obs(config, State)
Obsop    = obsop.Obsop(config, State, dict_obs, Model)
Obsop.process_obs()
Basis    = basis.Basis(config, State)
Basis.set_basis(...)                # builds Q, locations, hyperparams

inv.Inv_4Dvar(config=config, State=State, Model=Model,
              dict_obs=dict_obs, Obsop=Obsop, Basis=Basis, Bc=Bc)

diag.Diag(config, State)
```

For tiled / large-domain runs the `run_assimilation.prepare_process(...)`
helper takes the same building blocks and dispatches them across windows.

---

## 13. Notes for future work

- Every pipeline component is selected by a `super` string and merged with
  defaults; **adding a new model/basis/obsop variant** boils down to (i)
  adding a `XXX_FOO = dict(super='XXX_FOO', ...)` block to
  [config_default.py](mapping/src/config_default.py) and (ii) wiring the
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
