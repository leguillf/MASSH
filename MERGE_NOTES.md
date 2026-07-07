# Merge Notes — branch `Vardyn_generation`

> **Purpose of this file.** This document records the changes made on the
> `Vardyn_generation` branch so they can be merged into the other branch
> (the long-running development / integration branch). It is written to be
> handed to an AI coding agent together with both checkouts: the agent should
> read each change below, locate the corresponding code on the target branch,
> and re-apply the change there (the surrounding code may differ, so apply the
> *intent*, not a blind patch).
>
> **How to use this file (for the AI agent doing the merge):**
> 1. Read each change entry: the *What*, *Why*, *Files*, and *Details*.
> 2. Find the matching location on the target branch. Names/line numbers may
>    differ — match on the described structure, not on line numbers.
> 3. Re-apply the change, adapting to the target branch's surrounding code.
> 4. After each change, confirm `python -m py_compile` passes on the edited files.
> 5. Flag any conflict where the target branch already diverges from what is
>    described here, and ask the human before guessing.
>
> **This file is append-only and will accumulate every future change made on
> this branch.** New changes are added as new dated sections at the bottom.

---

## Reference repositories

Some changes are ports from a sibling repository used as the source of truth:

- **Source of behaviour:** `MASSH_val` (sibling checkout, typically at
  `../MASSH_val` relative to this repo).
- **This repo:** `MASSH_generation`.

When an entry says "ported from MASSH_val", the canonical implementation lives
in that repo and was adapted to fit this repo's classes.

---

## Change 2026-06-16 — Load / compute bathymetry depth gradients (`dH_dx`, `dH_dy`) in `Model_csw1l`

### What
Added the ability to obtain the depth-gradient fields `dH/dx` and `dH/dy`
alongside the depth field `H` in the shallow-water model class `Model_csw1l`,
either by reading them from the bathymetry NetCDF file or by computing them with
finite differences when they are not prescribed.

### Why
The model needs the spatial gradient of the bathymetry. Previously only `H` was
loaded; the gradient was unavailable. This mirrors the behaviour already present
in `MASSH_val` (class `Model_sw1l_jax`), so the two codebases stay consistent.

### Files changed
- `mapping/src/config_default.py`
- `mapping/src/mod.py`

### Details

**1. `mapping/src/config_default.py` — `MOD_CSW1L` dict, `name_var_H` entry.**

The `name_var_H` dict (variable names for the depth NetCDF file, used **only by
the `MOD_CSW1L` model block**) gained two optional keys, `dvar_dx` and
`dvar_dy`, used to name the depth-gradient variables in the file.

- Before:
  ```python
  name_var_H = {'lon':'','lat':'','var':''}, # Variable names for the depth netcdf file
  ```
- After:
  ```python
  name_var_H = {'lon':'','lat':'','var':'','dvar_dx':'','dvar_dy':''}, # Variable names for the depth netcdf file. 'dvar_dx' and 'dvar_dy' are to prescribe the depth gradient components; if left empty they are computed from 'var'.
  ```

> ⚠️ Merge note: there are **two** `name_var_H` definitions in
> `config_default.py` (one per model block). Only the one inside the
> **`MOD_CSW1L`** dict was changed. Do not touch the other one unless the target
> branch intends the same feature there.

**2. `mapping/src/mod.py` — `Model_csw1l.__init__`, bathymetry-loading block.**

Inside the `if config.MOD.file_H_aux is not None and os.path.exists(...)` branch,
immediately after `self.H = grid.interp2d(...)` is computed:

- Defined a local helper `compute_grad(field)` that returns the `(grad_x, grad_y)`
  finite-difference gradient of `field` over `State.X` / `State.Y` (centered in
  the interior, one-sided at the edges). This is ported (adapted to use
  `State.X`/`State.Y`) from `Model_sw1l_jax` in `MASSH_val/mapping/src/mod.py`.
- If `name_var_H['dvar_dx']` and `name_var_H['dvar_dy']` are both non-empty, the
  two gradient fields are read and interpolated to the state grid the **same way
  as `H`**, by calling `grid.interp2d` with a copy of `name_var_H` whose `'var'`
  key is overridden with the gradient variable name:
  ```python
  self.dH_dx = grid.interp2d(ds, dict(config.MOD.name_var_H, var=name_dvar_dx), State.lon, State.lat)
  self.dH_dy = grid.interp2d(ds, dict(config.MOD.name_var_H, var=name_dvar_dy), State.lon, State.lat)
  ```
- Otherwise, the gradients are computed from `self.H`:
  ```python
  self.dH_dx, self.dH_dy = compute_grad(self.H)
  ```
- In the `else` branch (no bathymetry file; `self.H = config.MOD.H`, a scalar
  mean depth), `self.dH_dx` and `self.dH_dy` are set to zero arrays of shape
  `(State.ny, State.nx)` so the attributes always exist.

### Resulting public attributes
- `self.dH_dx` — depth gradient along x, shape `(ny, nx)`.
- `self.dH_dy` — depth gradient along y, shape `(ny, nx)`.

### Naming differences vs. MASSH_val (important for merging)
- `MASSH_val`'s `Model_sw1l_jax` distinguishes `bathymetry` (negative) from `H`
  (positive) and exposes `grad_bathymetry_x/y` and `grad_H_x/y`. This
  `Model_csw1l` class works only with `self.H`, so the new attributes are named
  `self.dH_dx` / `self.dH_dy`. If the target branch expects the
  `grad_H_x`/`grad_H_y` naming, rename accordingly when merging.

### Verification done
- `python -m py_compile mapping/src/mod.py mapping/src/config_default.py` passes.

### Not done / open questions
- Nothing in `MASSH_generation` consumes `self.dH_dx` / `self.dH_dy` yet (the
  model code in `mapping/models/model_sw1l/jswm.py` does not read them). If the
  target branch has a consumer, wire it up and confirm the expected attribute
  names.

---

## Change 2026-06-16 — Barotropic tidal velocity initialization in `Model_csw1l`

### What
Added the initialization of the barotropic tidal velocity fields to the
`Model_csw1l` class, ported (and trimmed) from `Model_sw1l_jax` in `MASSH_val`.
The model can now build per-timestep eastward/northward tidal velocity fields
either by computing them with **pyFES** from a tidal atlas, or by loading and
interpolating pre-computed velocity fields from a file.

### Why
Bring the tidal-velocity forcing available in `MASSH_val` into the generation
codebase's shallow-water model, keeping the two implementations consistent.

### Files changed
- `mapping/src/mod.py`
- `mapping/src/config_default.py`

### Details

**1. `mapping/src/mod.py` — `Model_csw1l`.**

- In `__init__`, after the time-axis (`self.timestamps`, `self.T`) is built and
  before the model-state initialization, added a call:
  ```python
  # Barotropic tidal velocity
  self.init_tidal_velocity(config,State)
  ```
- Added two new methods to the class:
  - `init_tidal_velocity(self, config, State)` — two mutually exclusive branches:
    1. `if config.MOD.path_tidal_model is not None:` → compute tidal velocity with
       **pyFES** (`import pyfes` is done lazily inside this branch). Writes the
       eastward/northward pyFES YAML configs under `config.EXP.tmp_DA_path`,
       evaluates the tide over `self.timestamps` on the model grid, converts
       cm/s→m/s, and stores `self.u_bar_data` / `self.v_bar_data` as
       `{t: field}` dicts keyed by `self.T`.
    2. `elif config.MOD.path_tidal_velocity is not None:` → load U/V velocity
       fields from file via `open_interpolate`, time-interpolate them onto
       `self.timestamps`, and store `self.u_bar_data` / `self.v_bar_data`.
       The debug plot is guarded by `if config.EXP.flag_plot>0:`.
    - Each branch prints an informative "(this can take a while)" message.
  - `open_interpolate(self, State, path, name_var, comp)` — opens (single file or
    multi-file/glob), longitude-convention-aligns, time-subsets, and spatially
    interpolates onto the model grid (`scipy.interpolate.griddata` imported
    locally). Supports 1D and 2D coordinate variables and single/multiple
    variable keys.

**2. `mapping/src/config_default.py` — `MOD_CSW1L` dict.**

Added, in a new "Barotropic tide velocity" sub-section after `w_waves`:
```python
path_tidal_model = None,        # if not None, tidal velocities are computed with pyFES
path_tidal_velocity = None,     # else, read tidal velocity fields from this path
name_var_tidal_velocity = None, # name of variables in path_tidal_velocity
```

### Differences vs. MASSH_val source (important for merging)
This is intentionally a **reduced** port of MASSH_val's `init_tidal_velocity`:

- **Dropped the prescribed-tidal-model (amplitude/phase) branch.** MASSH_val has
  a third branch (`elif config.MOD.path_tidal_model is not None:`) that reads
  per-constituent amplitude/phase fields into `tidal_Ua/Ug/Va/Vg` and uses
  `get_freq_and_phase` + `compute_tidal_velocity`. That branch and both helper
  methods were **not** ported. Consequently the pyFES branch here is gated on
  `path_tidal_model is not None` **alone** (MASSH_val also required
  `compute_pyfes == True`).
- **Dropped `compute_pyfes`, `name_var_tidal_model`, `w_names` config keys** —
  none are referenced by this reduced port. (MASSH_val keeps them.) In
  particular `self.omega_names` is **not** set in this class, because only the
  dropped branch used it.
- **Dropped the Gaussian smoothing** of the loaded velocity fields (MASSH_val's
  `smooth_wavelength` block); `config.MOD.smooth_wavelength` is not referenced.

> Merge guidance: if the target branch wants the full MASSH_val behaviour
> (amplitude/phase constituents, smoothing, `compute_pyfes` switch), re-introduce
> the dropped branch, helper methods, and config keys from
> `MASSH_val/mapping/src/mod.py` (`Model_sw1l_jax.init_tidal_velocity`,
> `get_freq_and_phase`, `compute_tidal_velocity`) rather than taking this reduced
> version.

### Verification done
- `python -m py_compile mapping/src/mod.py mapping/src/config_default.py` passes.
- Confirmed no dangling references to the dropped symbols (`compute_pyfes`,
  `name_var_tidal_model`, `w_names`, `omega_names`, `get_freq_and_phase`,
  `compute_tidal_velocity`, `tidal_Ua/Ug/Va/Vg`).

### Not done / open questions
- Nothing in `MASSH_generation` consumes `self.u_bar_data` / `self.v_bar_data`
  yet: the generation `self.swm = model(...)` constructor is not wired to receive
  tidal velocity (unlike MASSH_val's `Swm`, which takes `tidal_Ua=...` etc.), and
  the model step does not read these fields. This change only performs the
  initialization. Wiring the forcing into the model integration is still open.

---

## Change 2026-06-16 — Vertical-mode structure functions (phi) and internal-tide generation term in `Model_csw1l`

### What
Added, inline in `Model_csw1l.__init__`, the part of `Model_sw1l_jax.init_vertical_modes`
(from `MASSH_val`) that loads the vertical structure functions `phi_1_0`,
`phi_1_H`, `phi_0_H` from an auxiliary file and builds the internal-tide
generation term `self.generation`.

### Why
The internal-tide generation coefficient depends on the vertical-mode structure
functions, which were not loaded in the generation codebase's shallow-water
model. This brings that initialization over from `MASSH_val`, keeping the two
consistent.

### Files changed
- `mapping/src/mod.py`
- `mapping/src/config_default.py`

### Details

**1. `mapping/src/mod.py` — `Model_csw1l.__init__`.**

Inserted inline (not as a separate method) right after `self.mask = State.mask`
and before the `self.swm = model(...)` initialization:

- Initializes `self.phi_1_0 = self.phi_1_H = self.phi_0_H = None`.
- If `config.MOD.file_mode_aux` exists, opens it, applies the same inline
  longitude-convention handling as the existing `c`/`H` loading, and interpolates
  each of `phi_1_0`, `phi_1_H`, `phi_0_H` onto the model grid with
  `grid.interp2d` (one per-field `{'lon','lat','var'}` dict). Plots the three
  fields under `config.EXP.flag_plot>0`.
- Builds the generation term:
  ```python
  self.generation = -(self.c**2/(self.H))*(self.phi_1_0/self.g)*(self.phi_0_H*self.phi_1_H)
  ```
  then zeroes it on the sponge layers (`sponge_on_h_S/N/W/E`, guarded with
  `getattr(...)` exactly as in the source), and plots it under `flag_plot>0`.

**2. `mapping/src/config_default.py` — `MOD_CSW1L` dict.**

Added, in the phase-velocity sub-section after `name_var_c`:
```python
file_mode_aux = None,  # auxiliary file for the vertical structure functions phi_n(z)
name_var_mode = {'lon':'','lat':'','phi_1_0':'','phi_1_H':'','phi_0_H':''},
```

### Differences vs. MASSH_val source (important for merging)
This is a **partial** port of `init_vertical_modes`:

- **Phase velocity `c` block NOT copied.** In `Model_csw1l`, `c`, `Heb`, and the
  `cmin`/`cmax` clipping are already loaded/interpolated earlier in `__init__`
  (from `filec_aux`/`name_var_c`), so only the phi / generation part was ported.
- **`no_generation` switch NOT ported.** The source has a
  `if config.MOD.no_generation: self.generation = np.ones_like(...)` branch and a
  `no_generation` config key; both were intentionally omitted (no config key
  added, no `if` line in the code).
- **Smoothing block NOT ported** (the source's `smooth_wavelength` gaussian
  filter on `self.generation`).
- **Generation computation is guarded** behind a check that all three phi fields
  were loaded; `self.generation = None` otherwise. The source computes it
  unconditionally, which would crash on the default config (`file_mode_aux=None`)
  — so an unconditional copy would break every existing `Model_csw1l` run.
- **Final generation plot is guarded** by `config.EXP.flag_plot>0`; the source
  calls `plt.show()` unconditionally.
- The `He`-from-`Heb` fallback branch (`if self.c is None: ...`, `He_data`,
  `He_init`) was not ported — `self.c` is always set in `Model_csw1l`.

> Merge guidance: if the target branch wants the full MASSH_val behaviour
> (`no_generation`, smoothing, the `He`/`Heb` fallback), re-introduce them from
> `MASSH_val/mapping/src/mod.py` (`Model_sw1l_jax.init_vertical_modes`) and the
> matching config keys rather than taking this partial version.

### Verification done
- `python -m py_compile mapping/src/mod.py mapping/src/config_default.py` passes.
- Confirmed no `no_generation` references in either file.

### Not done / open questions
- `self.generation` (and the `phi_*` fields) are **not consumed** yet: the
  generation `self.swm = model(...)` constructor is not wired to receive
  `generation=...` / `phi_1_0=...` (unlike MASSH_val's `Swm`). This change only
  performs the initialization. Wiring it into the model is still open.

---

## Change 2026-06-16 — Thread `u_bar` / `v_bar` (barotropic tidal velocity) through the `Model_csw1l` step functions

### What
Plumbed the barotropic tidal velocity fields `u_bar` / `v_bar` through the JAX
time-stepping machinery of `Model_csw1l`, exactly the same way the balanced-motion
fields `u_bm` / `v_bm` are threaded. The values are pulled from
`self.u_bar_data` / `self.v_bar_data` (built by `init_tidal_velocity`).

### Why
Preparatory step so the SW model (`mapping/models/model_sw1l/jswm.py`) can later
use the tidal velocity in the dynamics. This change only carries the fields down
to `_jstep`; it does not yet use them.

### Files changed
- `mapping/src/mod.py` (class `Model_csw1l`)

### Details

- **Guard flag.** `init_tidal_velocity` now sets, at its top,
  `self.is_tidal_velocity = False` and `self.u_bar_data = self.v_bar_data = None`,
  and sets `self.is_tidal_velocity = True` at the end of each branch that
  populates the data (pyFES branch and `path_tidal_velocity` branch). This mirrors
  the existing `self.is_bm` flag and avoids an `AttributeError` when no tidal
  velocity is configured.
- **Retrieval** (added in `step`, `step_tgl`, `step_adj`, mirroring the `is_bm`
  block):
  ```python
  if self.is_tidal_velocity:
      u_bar = self.u_bar_data[t]
      v_bar = self.v_bar_data[t]
  else:
      u_bar = None
      v_bar = None
  ```
- **Signature/threading.** `u_bar, v_bar` were inserted **immediately after
  `u_bm, v_bm`** (before `nstep`) in every signature and call site:
  - `_jstep`, `_jstep_tgl`, `_jstep_adj` signatures.
  - The inner `wrapped_jstep` closures inside `_jstep_tgl` and `_jstep_adj` (so,
    like `u_bm`/`v_bm`, they are captured constants and **not** differentiated —
    they are not part of `primals`/`tangents`/`cotangents`).
  - All call sites: `step` → `_jstep_jit`; `step_tgl` → `_jstep_tgl_jit`;
    `step_adj` → both the forward `_jstep_jit` loop and the reverse
    `_jstep_adj_jit` loop.

> Merge guidance: in MASSH_val the equivalent fields may be named differently or
> wired into `Swm(...)` directly. When merging, match on the `u_bm`/`v_bm`
> threading pattern (same positions, same non-differentiated treatment) rather
> than on argument order alone.

### Verification done
- `python -m py_compile mapping/src/mod.py` passes.
- Verified every `_jstep*` signature and call site carries `u_bar, v_bar` right
  after `u_bm, v_bm`.

### Not done / open questions
- `u_bar` / `v_bar` now reach `_jstep` but are **not used** inside it yet, and are
  not passed to `self.swm` / the `swm_step_nstep` call. The user will specify how
  to use them in `mapping/models/model_sw1l/jswm.py` in a follow-up.

---

## Change 2026-06-16 — Internal-tide generation control parameter `itg` in `Model_csw1l`

### What
Added a new control parameter `itg` (internal-tide generation) to `Model_csw1l`,
analogous to the `ITG_COEFF` parameter of `Model_sw1l_jax` in `MASSH_val` (same
role, lower-case name `itg`). It is a `(ny, nx)` coefficient field, registered in
the control vector and threaded through the JAX forward / tangent / adjoint step
machinery. A validation block fails early when `itg` is requested without its
prerequisites.

### Why
Make the internal-tide generation coefficient an estimable control parameter in
the generation codebase's shallow-water model, consistent with `MASSH_val`.

### Files changed
- `mapping/src/mod.py` (class `Model_csw1l`)
- `mapping/src/config_default.py` (`MOD_CSW1L` `name_params` comment)

### Details

- **Naming.** The parameter key is the lower-case string `'itg'` (val uses upper
  case `'ITG_COEFF'`). Local variables: `itg`, `ditg` (tangent), `ad_itg`
  (adjoint).
- **Registration** in `__init__`:
  ```python
  if 'itg' in self.name_params:
      State.params['itg'] = np.zeros((self.ny, self.nx))
  ```
- **Threading as a differentiated control parameter** (like `alpha_He`, NOT like
  the passthrough `u_bm`/`u_bar`). `itg` was inserted **immediately after
  `h_WE`** (i.e. at the end of the differentiated group, before the
  non-differentiated `h_bm`) in:
  - `_jstep`, `_jstep_tgl`, `_jstep_adj` signatures.
  - The `wrapped_jstep` closures and the `primals` / `tangents` tuples of
    `_jstep_tgl` and `_jstep_adj` (so it participates in the JVP/VJP).
  - `_jstep_adj` unpacks the extra adjoint `_ad_itg`, accumulates
    `if ad_itg is not None: ad_itg += _ad_itg`, and returns it; `step_adj`
    writes it back to `adState.params['itg']`.
  - Retrieval in `step` (`itg`), `step_tgl` (`itg`, `ditg`), `step_adj` (`itg`,
    `ad_itg`), each guarded by `'itg' in self.name_params` (else `None`, which
    flows through as a `None` pytree leaf exactly like `alpha_He=None`).
- **Validation block** at the end of `__init__` (after the generation term is
  built, before `self.swm = model(...)`), guarded by `if 'itg' in self.name_params:`.
  Raises `ValueError` when:
  1. `config.MOD.file_mode_aux` is `None` or its path does not exist (no
     vertical-mode structure functions -> no generation term);
  2. both `config.MOD.path_tidal_model` and `config.MOD.path_tidal_velocity` are
     `None` (no barotropic tidal velocity);
  3. any required field is `None` or entirely NaN (`np.all(np.isnan(...))`),
     checked for: `generation`, `phi_1_0`, `phi_1_H`, `phi_0_H`, `dH_dx`,
     `dH_dy`, and a representative `u_bar` / `v_bar` timestep
     (`next(iter(self.u_bar_data.values()))`). NaN here usually signals an
     interpolation problem or wrong variable names in the auxiliary files.
- **Config.** Added `'itg'` to the `name_params` options comment in `MOD_CSW1L`.

### Differences vs. MASSH_val source (important for merging)
- This is the **`mod.py` plumbing only** (deliberate scope choice). `itg` is
  registered, threaded, and validated, but is **not yet used inside `_jstep`** and
  is **not passed to the swm model** — so the `rhs_itg` term is not computed and
  the `itg` adjoint is currently identically zero. In `MASSH_val`, `ITG_COEFF` is
  consumed in `mapping/models/model_sw1l/jswm.py`:
  ```python
  rhs_itg = (self.generation * (u_bar*grad_H_x + v_bar*grad_H_y)) * itg_coeff
  ```
- The "full NaN" check uses `np.all(np.isnan(...))` (entirely NaN). Switch to
  `np.any` if a stricter "contains any NaN" check is desired.

> Merge guidance: to complete the feature, pass `itg` (and `generation`,
> `dH_dx/dH_dy`, `u_bar/v_bar`) into the generation `self.swm = model(...)`
> constructor / step and add the `rhs_itg` term to the generation `jswm.py`,
> mirroring val's `ITG_COEFF` branch. See the related entries
> [[u_bar / v_bar threading]] and [[vertical-mode phi / generation]].

### Verification done
- `python -m py_compile mapping/src/mod.py mapping/src/config_default.py` passes.
- Verified every `_jstep*` signature, closure, `primals`/`tangents`/`cotangents`
  tuple, adjoint unpack/return, and call site carries `itg` consistently right
  after `h_WE`.

### Not done / open questions
- `itg` is plumbed but unused in the dynamics (see Differences). Wiring the
  `rhs_itg` forcing into `jswm.py`, and passing `itg`/`generation`/`u_bar`/`v_bar`
  to the model, is the remaining end-to-end work — to be done together with the
  `u_bar`/`v_bar` model usage.

---

## Change 2026-06-16 — Expose `grad_H_x` / `grad_H_y` / `generation` on the SW model for the `itg` forcing

### What
When `'itg'` is a controlled parameter, `Model_csw1l` now sets the three fields
required to compute the internal-tide generation forcing (`rhs_itg`) directly on
the SW model object `self.swm`. The SW model (`jswm.py`) declares these as
`None` attributes by default.

### Why
Val's `rhs_itg` (in `MASSH_val/mapping/models/model_sw1l/jswm.py`) reads
`self.generation`, `self.grad_H_x`, `self.grad_H_y` off the model object:
```python
rhs_itg = (self.generation * (u_bar*self.grad_H_x + v_bar*self.grad_H_y)) * itg_coeff
```
This change makes those fields available on the generation codebase's SW model so
that forcing can later be implemented. It is the next step after the `itg`
parameter plumbing.

### Files changed
- `mapping/src/mod.py` (class `Model_csw1l`)
- `mapping/models/model_sw1l/jswm.py` (`__init__`)

### Details

- **`mapping/src/mod.py`** — in `Model_csw1l.__init__`, immediately after the
  block that sets the sponge attributes on `self.swm` (and before
  `self.swm.bc_it_method = ...`):
  ```python
  if 'itg' in self.name_params:
      self.swm.grad_H_x = self.dH_dx
      self.swm.grad_H_y = self.dH_dy
      self.swm.generation = self.generation
  ```
  Note the source→target name mapping: `self.dH_dx` / `self.dH_dy` (loaded /
  computed depth gradients) become `self.swm.grad_H_x` / `self.swm.grad_H_y`;
  `self.generation` keeps its name.
- **`mapping/models/model_sw1l/jswm.py`** — in `Swm.__init__` (the generation SW
  model class), declared the defaults next to the existing sponge-BC defaults
  (`self.flag_sponge_bc = False`, `self.sponge_coef = 0.`):
  ```python
  # Internal-tide generation forcing (attributes set externally by wrapper
  # when 'itg' is a controlled parameter)
  self.grad_H_x = None
  self.grad_H_y = None
  self.generation = None
  ```

### Differences vs. MASSH_val source (important for merging)
- In `MASSH_val`, `grad_H_x` / `grad_H_y` / `generation` are passed as
  **constructor arguments** to `Swm(...)`. Here they are instead set as
  **attributes after construction** (the same convention the generation codebase
  already uses for the sponge-BC fields), because the generation `Swm.__init__`
  signature does not take them. When merging, match on this attribute-injection
  convention rather than expecting constructor kwargs.

### Verification done
- `python -m py_compile mapping/src/mod.py mapping/models/model_sw1l/jswm.py`
  passes.

### Not done / open questions
- The `rhs_itg` term itself is still **not computed** in the generation
  `jswm.py`: the attributes now exist (`grad_H_x`, `grad_H_y`, `generation`, plus
  the `u_bar`/`v_bar` and `itg` plumbing from the earlier entries), but the actual
  forcing equation has not been added to the model's RHS. That is the remaining
  end-to-end step.

---

## Change 2026-06-17 — Non-flat-bottom flag (`flag_nonflat_bottom`) and its fields in `Model_csw1l` / SW model

### What
Ported the `flag_nonflat_bottom` parameter of `Mod_Sw1L` (`MASSH_val`) into the
generation codebase. When enabled, the SW model is meant to compute the spatial
derivatives in its equations **without** assuming a flat bottom — i.e. including
the spatial derivatives of the topography `H` and the surface first baroclinic
mode `phi_1(0)`. This change adds the flag, declares the fields it needs on the
SW model, and (in the wrapper) populates and validates those fields when the flag
is on. The dynamics themselves (the RHS branches) are **not** yet modified.

### Why
Bring `MASSH_val`'s non-flat-bottom SW formulation into `MASSH_generation`,
keeping the two consistent. Mirrors `Mod_Sw1L` / `MASSH_val/.../jswm.py`, where
`flag_nonflat_bottom` switches `rhs_u`/`rhs_v`/`rhs_h` between the flat-bottom
form and a form using `H`, `phi_1_0` and their u/v-grid interpolations.

### Files changed
- `mapping/models/model_sw1l/jswm.py` (`CSWm.__init__`)
- `mapping/src/mod.py` (class `Model_csw1l`)
- `mapping/src/config_default.py` (`MOD_CSW1L` dict)

### Details

**1. `mapping/models/model_sw1l/jswm.py` — `CSWm.__init__`.**

Declared default attributes next to the existing sponge-BC / itg-generation
defaults (set externally by the wrapper, same convention):
```python
self.flag_nonflat_bottom = False
self.H = None
self.H_u = None
self.H_v = None
self.phi_1_0 = None
self.phi_1_0_u = None
self.phi_1_0_v = None
```

**2. `mapping/src/config_default.py` — `MOD_CSW1L` dict.**

Added after `name_var_mode`:
```python
flag_nonflat_bottom = False, # if True, the spatial derivatives in the SW model equations are computed without considering that the bottom is flat, it includes the spatial derivatives of H and first mode at the surface
```

**3. `mapping/src/mod.py` — `Model_csw1l.__init__`.**

Immediately after the `itg` block that sets `grad_H_x`/`grad_H_y`/`generation`
on `self.swm` (and before `self.swm.bc_it_method = ...`), added a block guarded
by `if config.MOD.flag_nonflat_bottom:` that:
- Raises `ValueError` early if `self.H` or `self.phi_1_0` is `None` (H needs a
  bathymetry field; `phi_1_0` needs `config.MOD.file_mode_aux`).
- Sets the flag and the six fields on the SW model, computing the u/v-grid
  interpolations via the model's own `rho_on_u` / `rho_on_v`:
  ```python
  self.swm.flag_nonflat_bottom = True
  self.swm.H = self.H
  self.swm.H_u = self.swm.rho_on_u(self.H)
  self.swm.H_v = self.swm.rho_on_v(self.H)
  self.swm.phi_1_0 = self.phi_1_0
  self.swm.phi_1_0_u = self.swm.rho_on_u(self.phi_1_0)
  self.swm.phi_1_0_v = self.swm.rho_on_v(self.phi_1_0)
  ```
- Validates that the six populated fields are non-`None` and not entirely NaN
  (`np.all(np.isnan(np.asarray(field)))`), raising `ValueError` otherwise — same
  pattern as the `itg` validation. See [[itg control parameter]].

### Differences vs. MASSH_val source (important for merging)
- In `MASSH_val`, `flag_nonflat_bottom`, `H`, `phi_1_0` are **constructor
  arguments** to `Swm(...)`, and `H_u/H_v/phi_1_0_u/phi_1_0_v` are computed
  **inside** `jswm.__init__` via `rho_on_u`/`rho_on_v`. Here they are instead set
  as **attributes after construction**, and the u/v interpolations are computed
  **in the wrapper** — the same attribute-injection convention the generation
  codebase uses for the sponge-BC fields. Match on that convention when merging.
- The NaN check uses `np.all(np.isnan(...))` (entirely NaN). Switch to `np.any`
  for a stricter "contains any NaN" check.

### Verification done
- `python -m py_compile` passes on all three edited files (checked via `ast.parse`).

### Not done / open questions
- The RHS dynamics are **not** modified yet: `rhs_u` / `rhs_v` / `rhs_h` in the
  generation `jswm.py` still assume a flat bottom and do not branch on
  `self.flag_nonflat_bottom`. Porting the `if not self.flag_nonflat_bottom: ...
  else: ...` branches from `MASSH_val/.../jswm.py` (which divide by `phi_1_0` and
  scale by `H`, with `jnp.nan_to_num` guards) is the remaining end-to-end step.

---

## Change 2026-06-17 — Apply internal-tide generation forcing (`rhs_itg`) without controlling `itg` (`flag_itg`)

### What
Made the internal-tide generation forcing `rhs_itg` applicable in the SW model
**without** `itg` being a controlled parameter. Added a `flag_itg` switch: when
True the forcing is applied with a **null** `itg` coefficient (i.e. as if
`itg=0`). Adding `'itg'` to `name_params` implies `flag_itg=True`.

### Why
Previously `rhs_itg` was only ever computed when `'itg'` was in `name_params`
(the trigger in the SW model was `itg is not None`). The forcing itself does not
require `itg` to be estimated — the baseline term `generation*(u_bar*grad_H_x +
v_bar*grad_H_y)` is meaningful on its own (it is exactly the `itg=0` case of
`...*(1+itg)`). This lets the forcing be used as a fixed physical term.

### Files changed
- `mapping/src/config_default.py` (`MOD_CSW1L` dict)
- `mapping/models/model_sw1l/jswm.py` (`CSWm`)
- `mapping/src/mod.py` (class `Model_csw1l`)

### Details

**1. `mapping/src/config_default.py` — `MOD_CSW1L` dict.**
Added after `name_var_mode`:
```python
flag_itg = False, # if True, the internal-tide generation forcing (rhs_itg) is applied in the SW model even when 'itg' is not a controlled parameter (equivalent to a null itg coefficient). Adding 'itg' to name_params turns this on automatically.
```

**2. `mapping/models/model_sw1l/jswm.py` — `CSWm`.**
- Added default attribute `self.flag_itg = False` to the itg-generation block in
  `__init__` (set externally by the wrapper).
- `compute_rhs_itg` signature changed to `compute_rhs_itg(self, u_bar, v_bar, itg=None)`;
  `itg` is now optional. When `itg is None` the `(1+itg)` modulation is skipped, so
  `rhs_itg = generation*(u_bar*grad_H_x + v_bar*grad_H_y)`.
- In `_step_rk4_nstep` and `_step_euler_nstep`, the gating changed from
  `if itg is not None and u_bar is not None and v_bar is not None:` to
  `if self.flag_itg and u_bar is not None and v_bar is not None:`, and the call is
  now `self.compute_rhs_itg(u_bar, v_bar, itg)` (so the forcing is applied whenever
  `flag_itg`, with `itg` possibly `None`).

**3. `mapping/src/mod.py` — `Model_csw1l.__init__`.**
- After the `itg` param registration, compute the flag:
  ```python
  self.flag_itg = bool(getattr(config.MOD, 'flag_itg', False)) or ('itg' in self.name_params)
  ```
- The generation-forcing **validation block** guard changed from
  `if 'itg' in self.name_params:` to `if self.flag_itg:` (messages reworded to
  "the internal-tide generation forcing is requested (flag_itg or 'itg' in
  name_params)"). Requirements are unchanged (needs `file_mode_aux` and a tidal
  velocity field; fields non-`None` and not entirely NaN).
- The block that sets the forcing fields on the SW model changed from
  `if 'itg' in self.name_params:` to `if self.flag_itg:`, and now also sets
  `self.swm.flag_itg = True` (in addition to `grad_H_x`/`grad_H_y`/`generation`).
- `step`/`step_tgl`/`step_adj` are unchanged: `itg` is still pulled from
  `State.params` only when `'itg'` is controlled, otherwise `itg=None` (and
  `ditg`/`ad_itg=None`). When `flag_itg` is on but `'itg'` is not controlled, the
  `None` flows through and the SW model applies the null-coefficient forcing. Since
  that forcing is then a constant (no dependence on differentiated controls), it
  contributes nothing to the tangent/adjoint — `ad_itg` stays `None` and is not
  written back. See [[itg control parameter]].

### Differences vs. MASSH_val source (important for merging)
- This is generation-codebase-specific plumbing; `MASSH_val` gates the forcing
  differently (via `ITG_COEFF`). Match on the `flag_itg` semantics: "forcing
  applied iff flag_itg; itg coefficient optional, null when absent".

### Verification done
- `python -m py_compile` passes on all three edited files (checked via `ast.parse`).

### Not done / open questions
- None specific. The forcing now works both as a fixed term (`flag_itg=True`,
  `'itg'` not controlled) and as a controlled-parameter modulation (`'itg'` in
  `name_params`).

---

## Change 2026-06-17 — Implement the `flag_nonflat_bottom` effect in the SW RHS (`rhs_u`/`rhs_v`/`rhs_h`)

### What
Ported the actual non-flat-bottom formulation of `rhs_u`, `rhs_v` and `rhs_h`
from `MASSH_val/.../jswm.py` into the generation SW model (`CSWm`). Each RHS now
branches on `self.flag_nonflat_bottom`: the flat-bottom form is unchanged; the
non-flat form uses the topography `H` and the surface first-mode `phi_1(0)` (and
their u/v-grid interpolations) declared/populated in the earlier entries.

### Why
Completes the non-flat-bottom feature: the earlier entry only declared the flag
and fields and populated them from the wrapper; the dynamics still assumed a flat
bottom. This adds the physics so enabling `config.MOD.flag_nonflat_bottom`
actually changes the equations, matching `MASSH_val`.

### Files changed
- `mapping/models/model_sw1l/jswm.py` (`CSWm.rhs_u`, `rhs_v`, `rhs_h`)

### Details
Adapted the `if not self.flag_nonflat_bottom: <flat> else: <non-flat>` branches
to the generation grid conventions (`u`:(ny,nx-1), `v`:(ny-1,nx), `h`:(ny,nx),
precomputed `DX`/`DY`/`DXu`/`DYv`), rather than copying val's full-grid inline
coordinate differences verbatim.

- **`rhs_u`** — non-flat pressure gradient:
  ```python
  h_phi_1_0 = h / self.phi_1_0
  rhs_u[1:-1,:] = f_on_u·v_on_u(v)
                  - g * phi_1_0_u[1:-1,:] * (h_phi_1_0[1:-1,1:]-h_phi_1_0[1:-1,:-1]) / DX[1:-1,:]
  rhs_u = jnp.nan_to_num(rhs_u, 0.0)
  ```
- **`rhs_v`** — analogous on the v-grid, with `phi_1_0_v[:,1:-1]` and
  `(h_phi_1_0[1:,1:-1]-h_phi_1_0[:-1,1:-1]) / DY[:,1:-1]`, then `nan_to_num`.
- **`rhs_h`** — non-flat continuity:
  ```python
  Hu_phi_1_0 = H_u*u/phi_1_0_u ;  Hv_phi_1_0 = H_v*v/phi_1_0_v
  rhs_h[1:-1,1:-1] = -(He·phi_1_0/H)[1:-1,1:-1] * (
        (Hu_phi_1_0[1:-1,1:]-Hu_phi_1_0[1:-1,:-1]) / DXu[1:-1,:]
      + (Hv_phi_1_0[1:,1:-1]-Hv_phi_1_0[:-1,1:-1]) / DYv[:,1:-1])
  rhs_h = jnp.nan_to_num(rhs_h, 0.0)
  ```
  The `rhs_itg` add and the mean-flow advection/shear/divergence terms (`u11u`,
  `u11z`, `u11p`, …) are **outside** the branch and unchanged — they are applied
  identically in both bottom regimes (matching val, which also keeps them).

### Differences vs. MASSH_val source (important for merging)
- Same **physics**, different **indexing**: ported to generation's staggered
  shapes + precomputed metric arrays instead of val's `(X[..,2:-1]-X[..,1:-2])`
  inline differences. The `H_u/H_v/phi_1_0_u/phi_1_0_v` interpolations are computed
  in the wrapper here (see [[non-flat-bottom flag]]), not in `__init__`.
- `He` passed to `rhs_h` is the **total** equivalent depth (`Heb + He2d`), as in
  the flat branch and as in val.
- Kept val's `jnp.nan_to_num(..., nan=0.0)` guards on the non-flat branches (for
  cells where `H` or `phi_1_0` vanish, e.g. land). Note this can mask NaNs in the
  adjoint; acceptable as a faithful port, revisit if gradient issues appear.

### Verification done
- `python -m py_compile mapping/models/model_sw1l/jswm.py` passes (via `ast.parse`).
- Hand-checked array shapes of every sliced term in all three non-flat branches
  against the generation grid conventions; all conform.

### Not done / open questions
- No numerical/regression run performed (no suitable `H`/`phi_1_0` test fixture to
  hand). Recommend a forward run with `flag_nonflat_bottom=True` vs `False` on a
  case with non-trivial topography to confirm the expected difference, plus the
  4DVar tangent/adjoint tests given the `nan_to_num` non-smoothness.

---

## Change 2026-06-17 — New `tools_video.py` helper to generate videos from a matplotlib plotting function

### What
Added a new standalone module `mapping/src/tools_video.py` providing a single
helper, `generate_video(...)`, that renders a sequence of frames with a
user-supplied matplotlib plotting function and assembles them into an `.mp4`
video with `ffmpeg`.

### Why
Convenience utility for producing animations (e.g. of model fields over time)
from any per-frame plotting callback, with parallel frame rendering.

### Files changed
- `mapping/src/tools_video.py` (new file)

### Details
`generate_video(func, inputs, output_dir, fps=12, n_jobs=10, delete_frames=True, video_name="movie.mp4")`:
- Creates `output_dir` and a `frames/` subdirectory, clearing any pre-existing
  `.png` frames there.
- For each `item` in `inputs`, calls the user callback as
  `func(item, filename)` where `filename` is `frames/{i:06d}.png` — i.e. the
  plotting function is responsible for drawing **and saving** that frame.
- Renders frames in parallel via `joblib.Parallel(n_jobs=..., verbose=10)`.
- Assembles the frames into `output_dir/video_name` by invoking `ffmpeg`
  (`-framerate fps -i %06d.png -c:v libx264 -pix_fmt yuv420p`).
- Deletes the `frames/` directory afterwards unless `delete_frames=False`.

### Dependencies / environment
- Requires `ffmpeg` available on `PATH` (called via `subprocess.run(..., check=True)`).
- Requires `joblib`. The plotting callback supplies its own matplotlib usage.

### Verification done
- N/A (new standalone helper; no other module imports it yet).

### Not done / open questions
- Nothing in the codebase calls `generate_video` yet; it is a utility to be wired
  into diagnostics/plotting where needed.

---

## Change 2026-06-23 — Apply a prescribed control vector through a reduced basis in `Inv_forward`

### What
Extended `Inv` / `Inv_forward` (in `mapping/src/inv.py`) so a forward-only run
(`config.INV is None`) can project a prescribed control vector `X` onto the model
parameters at every saved-output step through a reduced `Basis`. Previously
`Inv_forward` only ran a bare forward integration with no way to inject a control
vector / reduced basis.

### Why
Allows re-running a forward integration that reproduces the parameter trajectory
of a 4Dvar analysis (or any prescribed control vector), e.g. by feeding the
`Xres.nc` control vector written by the analysis. The basis is evaluated in time
so time-dependent parameters (`He_mean`, `hbcx`/`hbcy` entering the waves, …) are
reconstructed step by step.

### Files changed
- `mapping/src/inv.py` (`Inv`, `Inv_forward`)

### Details

**1. `Inv` signature / dispatch.**
- Added an `X=None` argument to `Inv(...)`.
- The forward-only branch now forwards `Basis` and `X`:
  ```python
  if config.INV is None:
      return Inv_forward(config, State=State, Model=Model, Basis=Basis, X=X, Bc=Bc)
  ```

**2. `Inv_forward` signature.**
- Changed to `Inv_forward(config, State, Model, Basis=None, X=None, Bc=None)`.
- `X` is expected to be the **full, already-projected** control vector (the
  content of the `Xres.nc` file from the 4Dvar analysis); it is passed to
  `Basis.operg` as-is — no background or preconditioning is re-applied here.

**3. Basis setup (before the time loop).**
- The basis is evaluated at every saved-output node, in **days**, because
  `operg` can only be called at the exact time nodes that are the keys of `Gt`:
  ```python
  time_basis = np.arange(0, Model.T[-1] + nstep*Model.dt, nstep*Model.dt) / 24 / 3600
  Xb, _ = Basis.set_basis(time_basis, return_q=True, State=State)
  ```
- Control-vector handling:
  - `X is None` → use a null control vector `np.zeros((Xb.size,))` (prints a
    notice).
  - `X` provided → `np.asarray(X)`, and its size is validated against `Xb.size`
    (`sys.exit` with a descriptive message on mismatch).
  - `Basis is None` but `X` provided → warn that `X` is ignored.

**4. Projection inside the integration.**
- A step counter `it` is maintained alongside `t`.
- Right after `Model.init(State, t)` (before the first `save_output`), the basis
  is projected for the initial step:
  ```python
  if Basis is not None:
      Basis.operg(time_basis[it], X, State=State.params)
  ```
- After each propagation step (`it += 1`), the time-dependent control vector is
  re-projected onto the model parameters, guarded against running past the last
  node:
  ```python
  if Basis is not None and it < time_basis.size:
      Basis.operg(time_basis[it], X, State=State.params)
  ```

### Differences vs. MASSH_val source (important for merging)
- This is generation-codebase plumbing on the forward-only path. When merging,
  match on the intent: evaluate the basis on the saved-output time nodes (in
  days), project `X` via `operg(t, X, State=State.params)` at the initial step and
  after every step. The `Basis.set_basis(..., return_q=True, State=State)` /
  `Basis.operg(...)` API is whatever the target branch's basis class exposes.

### Verification done
- `python -m py_compile mapping/src/inv.py` passes.

### Not done / open questions
- No end-to-end run performed (depends on a valid reduced `Basis` and a matching
  `Xres.nc` control vector). Recommend a forward run with a real basis + control
  vector to confirm the reconstructed parameter trajectory matches the analysis.

---

## Change 2026-06-23 — Align BM dataset longitude convention to `State.lon` in `_compute_bm_fields`

### What
In `Model_csw1l._compute_bm_fields` (the method that loads the balanced-motion
"BM" dataset, interpolates it onto the model grid, and derives the geostrophic
velocities), the BM dataset's longitude coordinate is now converted to the same
convention as `State.lon` **before** any spatial subsetting / interpolation.

### Why
The BM dataset was opened and fed straight into `sel`/`interp`/`griddata`
against `State.lon` without first reconciling the longitude convention
(`-180..180` vs `0..360`). If the BM file used the opposite convention to the
model state, the subdomain selection and interpolation silently produced empty
or NaN fields. Every other dataset loader in `mod.py` (the `c`, `mdt`, `bathy`,
`H`, `mode` loaders) already performs this same conversion; this brings the BM
loader in line.

### Files changed
- `mapping/src/mod.py` (`Model_csw1l._compute_bm_fields`)

### Details
Immediately after `dsbm = xr.open_mfdataset(path_bm, chunks=None)` and the
`name_lon_bm` / `name_lat_bm` lookups, added the same longitude-convention block
used elsewhere in the file:
```python
# Adapt longitude convention to match State.lon
lon = dsbm[name_lon_bm]
if np.sign(lon.data.min())==-1 and State.lon_unit=='0_360':
    dsbm = dsbm.assign_coords({name_lon_bm:((name_lon_bm, lon.data % 360))})
elif np.sign(lon.data.min())>=0 and State.lon_unit=='-180_180':
    dsbm = dsbm.assign_coords({name_lon_bm:((name_lon_bm, (lon.data + 180) % 360 - 180))})
```
This runs before the time-range selection and before both the 1D (`sel` +
`interp`) and 2D (`griddata`) interpolation paths, so all downstream comparisons
against `State.lon` are in a consistent convention.

### Differences vs. MASSH_val source (important for merging)
- This is the identical pattern already used by the other dataset loaders in this
  repo (e.g. the `c`/`mdt`/`bathy` blocks around the `name_var_*['lon']` lookups).
  Match on that pattern when merging rather than on line numbers.

### Verification done
- `python -m py_compile mapping/src/mod.py` passes.

### Not done / open questions
- The conversion uses `lon.data.min()` (the existing repo idiom), which assumes
  the longitude is a 1D coordinate-style array. It still works for the 2D-coord
  branch since `assign_coords` reassigns same-shaped data, but if a BM file stores
  `lon` as a 2D **non-dimension** coordinate this may need a different assignment.

---

## Change 2026-07-01 — Open-boundary sponge extension (`extend_it_open_boundary_sponge`) in `Model_csw1l`

### What
Ported the internal-tide open-boundary sponge *extension* feature from the
`VarDyn` sibling checkout (`../VarDyn`, class `Model_csw1l`). When enabled, the
IT-side total equivalent depth (`Heb`, `He_mean` and the `alpha_He` coupling
control) is extrapolated from the *sponge interior edge* outward across the
open-boundary S/N/W/E sponge bands, using a smooth (quintic smoothstep) S/N/W/E
corner partition of unity, before the total equivalent-depth medium `He_total`
is handed to the SW model. The domain bathymetry `H` is extended the same way
and stored as `self.H_it_open_boundary`.

### Why
Inside the open-boundary sponge bands the control-parameter media are otherwise
poorly constrained / discontinuous with the interior, which distorts the
entering-wave / generation medium. Extending the interior-edge values across the
band gives a smoother, physically consistent `He_total` for the boundary
treatment. The extension is a pure pre-processing of the medium; it is
behaviour-preserving when disabled.

### Files changed
- `mapping/src/mod.py` — new `_ITOpenBoundaryExtensionMixin`; `Model_csw1l` now
  inherits it; `__init__`, `_compute_it_open_boundary_He_total`, and `_jstep`.
- `mapping/src/config_default.py` — new `MOD_CSW1L` params
  `bc_it_corner_weight_power` and `extend_it_open_boundary_sponge`.

### Details

**1. Config (`MOD_CSW1L`).** Added, after `bc_it_method`:
```python
bc_it_corner_weight_power = 1.0,       # power on the S/N/W/E corner partition weights
extend_it_open_boundary_sponge = False,# enable the extension (default off)
```

**2. `_ITOpenBoundaryExtensionMixin`** (new class, inserted before `class M:`,
copied verbatim from `VarDyn`). Provides:
- `_init_it_open_boundary_extension(config_mod)` — precomputes, on h-points, the
  normalized S/N/W/E smoothstep weights and the sponge-interior-edge row/column
  indices (`iS`/`iN`/`jW`/`jE`). No-ops (stays inactive) unless
  `extend_it_open_boundary_sponge` is set, `flag_bc_sponge` is true, the four
  `sponge_on_h_{S,N,W,E}` masks exist, and at least one is non-empty. Only
  open-boundary bands are handled; coast/island extension is deliberately left
  for a future step.
- `_extend_it_open_boundary_field(field)` — jax path used inside `_jstep`;
  returns the field unchanged when inactive (so it is jit- and autodiff-safe).
- `_extend_it_open_boundary_static_field(field)` — numpy wrapper for static
  fields (used for `H`).

**3. `Model_csw1l.__init__`.** Right after `self.mask = State.mask` (and after
the sponge masks are built):
```python
self._init_it_open_boundary_extension(config.MOD)
self.H_it_open_boundary = self._extend_it_open_boundary_static_field(self.H)
```

**4. `_compute_it_open_boundary_He_total(He, alpha_He, h_bm)`** (new method next
to `_compute_He_from_bm`): extends `Heb`, `He_mean` and `alpha_He`, then returns
`Heb_ext + _compute_He_from_bm(He_ext, alpha_He_ext, h_bm)`.

**5. `_jstep`.** The `He_total` passed to `swm_step_nstep` changed from
`self.Heb+He2d` to `self._compute_it_open_boundary_He_total(He_mean, alpha_He, h_bm)`.
The separate `He2d` (still `self.Heb+He2d` in the `w1ext` branch) is unchanged;
`w1ext` is only computed when `not flag_bc_sponge`, i.e. never when the extension
is active.

### Differences vs. VarDyn source (important for merging)
- VarDyn also mixes `_ITOpenBoundaryExtensionMixin` into `Model_bmit`. That was
  **not** ported here — this change is scoped to `Model_csw1l` per request. The
  mixin and the `MOD_BMIT` config are left untouched.
- VarDyn additionally sets `self.swm.bc_it_corner_weight_power = ...` on the SW
  model. That was **not** ported: this repo's `jswm.py` has no such attribute
  (it would `AttributeError`); `bc_it_corner_weight_power` is instead consumed
  only inside the mixin via `getattr(..., 1.0)`, which is self-contained.
- Tangent-linear / adjoint need no separate change: `_jstep_tgl` / `_jstep_adj`
  wrap `_jstep` through `jax.jvp` / `jax.vjp`, so the extension differentiates
  automatically.

### Verification done
- `python -m py_compile mapping/src/mod.py mapping/src/config_default.py` passes.
- Behaviour-preserving with the default `extend_it_open_boundary_sponge=False`:
  `_extend_it_open_boundary_field` returns inputs unchanged, so `He_total`
  reduces exactly to the previous `self.Heb+He2d`.

### Not done / open questions
- Not applied to `Model_bmit` (out of the requested scope).
- No end-to-end run with the flag enabled; recommend a forward run over a domain
  with an active open-boundary sponge to confirm the extended `He_total` medium.

---

## Change 2026-07-01 — Smooth S/N/W/E corner blending of IT sponge fields (`bc_it_corner_weight_power`) in the SW model

### What
Ported, from `VarDyn`'s `mapping/models/model_sw1l/jswm.py`, the smooth
partition-of-unity blending used to combine the S/N/W/E internal-tide wave
fields inside `compute_IT_2D`, together with its control attribute
`bc_it_corner_weight_power`. This replaces the previous boolean sponge-mask
multiplication + integer-count averaging (`/ weight_sponge_*`), which produced
hard jumps where two sponge bands overlap (the corners).

### Why
Where two open-boundary sponge bands meet, the old `mask * field` accumulation
divided by an integer count (2 in the corner, 1 on an edge) gives a
discontinuous blend. A smooth (quintic smootherstep) partition of unity removes
those corner jumps. `bc_it_corner_weight_power` is an optional exponent on the
per-edge weights that biases the blend toward the nearest boundary (1.0 = plain
smootherstep, the default and previous corner behaviour aside from the
discontinuity fix). This is the SW-model counterpart consumed by the
`extend_it_open_boundary_sponge` feature (see the previous entry).

### Files changed
- `mapping/models/model_sw1l/jswm.py` — new `bc_it_corner_weight_power` attr;
  new `_smootherstep`, `_edge_weight`, `_it_boundary_weights` helpers; reworked
  `compute_IT_2D` blending.
- `mapping/src/mod.py` (`Model_csw1l.__init__`) — pushes the config value onto
  the SW model.
- (`bc_it_corner_weight_power` was already added to `MOD_CSW1L` in
  `mapping/src/config_default.py` in the previous change.)

### Details

**1. `jswm.__init__`.** Added next to `self.bc_it_method`:
```python
self.bc_it_corner_weight_power = 1.0
```

**2. New helpers (copied verbatim from VarDyn), inserted before `compute_IT_2D`.**
- `_smootherstep(x)` — quintic `6x^5-15x^4+10x^3`.
- `_edge_weight(dist, mask)` — `1 - smootherstep(dist/width)` inside `mask`,
  0 outside; raised to `self.bc_it_corner_weight_power` when it is not 1.0. The
  `!= 1.0` test is a static Python branch (the attr is a plain float), so it is
  jit-safe.
- `_it_boundary_weights(grid)` — builds the four normalized S/N/W/E weights on
  the requested grid (`h` / `u` / `v`) from that grid's `sponge_on_*` masks and
  coordinates. Falls back to a uniform `0.25` split if the masks are absent.

**3. `compute_IT_2D`.** Three changes:
- Compute `wh_*`, `wu_*`, `wv_*` once via `_it_boundary_weights` after `He_on_v`.
- Drop the `self.sponge_on_{h,u,v}_{S,N,W,E} *` factors from the per-direction
  accumulations (the smooth weights now localize the fields).
- Replace `(f_S + f_N + f_W + f_E) / self.weight_sponge_*` with
  `w_S*f_S + w_N*f_N + w_W*f_W + w_E*f_E`.

**4. `Model_csw1l.__init__`.** After setting `self.swm.bc_it_method`:
```python
self.swm.bc_it_corner_weight_power = getattr(
    config.MOD, 'bc_it_corner_weight_power', self.swm.bc_it_corner_weight_power)
```

### Differences vs. VarDyn source / notes for merging
- `weight_sponge_u/v/h` are still computed in `Model_csw1l` and set on the SW
  model, but are no longer read by `compute_IT_2D`. Left in place (harmless) to
  avoid touching unrelated code; can be removed later if confirmed unused.
- Only `Model_csw1l` pushes the attribute onto its SW model. VarDyn also does so
  from `Model_bmit`; not ported here (out of scope — matches the previous entry).
- The SW model's `compute_IT_2D` is shared, so `Model_bmit` would automatically
  pick up the new smooth blending with the default power 1.0.

### Verification done
- `python -m py_compile mapping/models/model_sw1l/jswm.py mapping/src/mod.py
  mapping/src/config_default.py` passes.
- The ported helpers + `compute_IT_2D` blending are byte-identical to VarDyn
  aside from the docstring/comment wording (verified by `diff`).

### Not done / open questions
- Numerically changes existing sponge runs even at the default power (corner
  cells now blend smoothly instead of a hard 1/2 average). Recommend a
  before/after forward run over a domain with an active sponge to confirm the
  boundary fields look right.
- No pruning of the now-unused `weight_sponge_*` plumbing.

---

## Change 2026-07-01 — Make the open-boundary sponge extension NaN-tolerant in `_extend_it_open_boundary_field`

### What
Reworked `_ITOpenBoundaryExtensionMixin._extend_it_open_boundary_field` (in
`mapping/src/mod.py`) so NaNs in a field no longer block its extension across the
open-boundary sponge bands. This affects every field that goes through the
extension: the static media `H`, `c`, and the vertical-mode structure functions
`phi_1_0` / `phi_1_H` / `phi_0_H`, and the dynamic `Heb` / `He` / `alpha_He`
(all route through this one method — see [[open-boundary sponge extension]]).

### Why
The extension fills each sponge band by blending the four interior-edge reference
rows/columns (`refS/refN/refW/refE`, taken at the sponge-interior indices
`iS/iN/jW/jE`) with a smooth S/N/W/E partition of unity. Previously the reference
row/column was broadcast **as-is**: if it contained a NaN (land, or an
interpolation gap in `c`/`H`/`phi`), that NaN propagated into the weighted sum and
the sponge point stayed NaN — i.e. the value was *not* extended over NaNs. The
user wants the value extended over NaNs.

### Files changed
- `mapping/src/mod.py` (`_ITOpenBoundaryExtensionMixin`)

### Details
Two complementary, jit-/autodiff-safe changes (the method is called both on
static numpy fields via `_extend_it_open_boundary_static_field` and inside the
traced `_jstep` on `Heb`/`He`/`alpha_He`, so everything must stay jax-friendly —
no data-dependent shapes or boolean indexing):

**1. New static helper `_fill_nan_along_axis(line, axis)`.** Fills NaNs *within*
each interior-edge reference row/column by the nearest valid value along the line
(forward fill then backward fill), implemented with `jax.lax.cummax` /
`jax.lax.cummin` over broadcast index arrays plus `jnp.take_along_axis`. A line
position with no finite value anywhere along `axis` is left NaN.
- `refS`/`refN` (rows) are filled along `axis=1`; `refW`/`refE` (columns) along
  `axis=0`, before being broadcast to `field.shape`.

**2. NaN-aware weighted blend.** After broadcasting, any reference still NaN is
dropped and the smooth weights are renormalized over the remaining valid
references:
```python
valid  = [jnp.isfinite(r) for r in refs]
eff_w  = [jnp.where(v, w, 0.0) for v, w in zip(valid, wref)]
safe_r = [jnp.where(v, r, 0.0) for v, r in zip(valid, refs)]
wsum      = eff_w[0]+eff_w[1]+eff_w[2]+eff_w[3]
wsum_safe = jnp.where(wsum > 0, wsum, 1.0)
extended  = (Σ eff_w*safe_r) / wsum_safe
extended  = jnp.where(wsum > 0, extended, field)   # no valid ref anywhere -> keep original
```
So a sponge point is filled as long as at least one of its active directional
references is finite; if none are, the original value is preserved (rather than
being turned into a spurious 0).

### Differences vs. VarDyn source (important for merging)
- This is a fix *on top of* the ported extension (see the 2026-07-01
  "Open-boundary sponge extension" entry). If VarDyn's
  `_extend_it_open_boundary_field` still does the plain broadcast + weighted sum,
  re-apply this NaN handling there too. Match on the *intent* (fill NaNs along
  each reference line, then renormalize the blend over finite references), not on
  a line patch.

### Verification done
- `python -m py_compile mapping/src/mod.py` passes (via `ast.parse`).
- Unit-checked `_fill_nan_along_axis` under jax: `[nan,1,nan,3,nan]` → `[1,1,1,3,3]`
  (row, axis=1); `[nan,2,nan,nan,5]` → `[2,2,2,2,5]` (column, axis=0); an
  all-NaN line stays NaN.

### Not done / open questions
- No end-to-end run; recommend a forward run with
  `extend_it_open_boundary_sponge=True` on a domain whose `c`/`H`/`phi` contain
  NaNs (land) to confirm the sponge bands are now filled as expected.

---

## Change 2026-07-02 — Port the newest vectorised entering-IT computation into `Model_csw1l._compute_IT_2D` (`mod.py`)

### What
Replaced the old nested `omega × theta` Python-loop implementation of
`Model_csw1l._compute_IT_2D` (in `mapping/src/mod.py`) with the newest vectorised
version of the entering-internal-wave computation from the `VarDyn` sibling
checkout (`../VarDyn`). In `VarDyn` this computation lives in the SW model
(`mapping/models/model_sw1l/jswm.py`, `compute_IT_2D`); it was already ported to
this repo's `jswm.py` (see the two 2026-07-01 entries). This change brings the
**stand-alone `mod.py` copy** — `Model_csw1l._compute_IT_2D`, a jit-compiled but
otherwise self-contained duplicate used for diagnostics — up to the same
algorithm so the two no longer diverge.

### Why
`Model_csw1l._compute_IT_2D` was still the old formulation: a double Python loop
over `omegas × bc_theta`, boolean `sponge_on_*` mask multiplication per
direction, and a final `/ weight_sponge_*` integer-count average that produces
hard corner jumps where two sponge bands overlap. The newest version (a) is
vectorised over `theta` (O(1) XLA nodes instead of unrolling the theta loop),
(b) supports the `plane_wave` / `plane_wave_bdy` / `wkb` phase methods via
`bc_it_method`, and (c) blends the S/N/W/E fields with a smooth quintic
partition of unity instead of the discontinuous integer-count average.

### Files changed
- `mapping/src/mod.py` (class `Model_csw1l`)

### Details

**1. Four helper methods added to `Model_csw1l`** (copied from this repo's
`jswm.py` `CSWm`, adapted — see below), inserted just before `_compute_IT_2D`:
- `_wave_phases(w, He, He_on_u, He_on_v)` — per-boundary phase/amp/kx/ky for all
  `bc_theta` at once, on the h/u/v grids; branches on `self.bc_it_method`.
- `_smootherstep(x)` — quintic `6x^5-15x^4+10x^3`.
- `_edge_weight(dist, mask)` — `1 - smootherstep(dist/width)` inside `mask`,
  raised to `self.bc_it_corner_weight_power` when != 1.0.
- `_it_boundary_weights(grid)` — normalised smooth S/N/W/E partition of unity on
  the requested grid; uniform `0.25` fallback if masks are absent.

**2. `_compute_IT_2D` body** replaced with the vectorised accumulation +
`w_S*f_S + w_N*f_N + w_W*f_W + w_E*f_E` blend (drops the `sponge_on_*` factors
and the `/ weight_sponge_*` average).

**3. `__init__`** — `self.bc_it_method` and `self.bc_it_corner_weight_power` are
now set on the model instance (previously only pushed onto `self.swm`), because
the ported helpers read them off `self`. The existing `self.swm.*` assignments
are preserved (now sourced from the `self.*` values).

### Differences vs. source (important for merging)
- **Grid-name adaptation.** `VarDyn`/`jswm` use `self.X` / `self.Y` for the
  h-grid; `Model_csw1l` uses `self.Xh` / `self.Yh`. The h-grid entries in
  `_wave_phases` and `_it_boundary_weights` were renamed accordingly. The u/v
  grids (`Xu/Yu/Xv/Yv`) match and are unchanged.
- **Method name.** Kept the leading-underscore name `_compute_IT_2D` (the class's
  existing API, jit-wrapped at `self._compute_IT_2D_jit`), vs. `jswm`'s public
  `compute_IT_2D`.
- **Scope.** Only `Model_csw1l` was updated per request. The **identical** old
  `_compute_IT_2D` in `Model_bmit` (same file) was intentionally left untouched.
- Aside from those renames, ASCII-ised comment glyphs, and docstring wording, the
  computational code is byte-identical to this repo's tested `jswm.py`
  `compute_IT_2D` / helpers (verified by normalised `diff`).
- `weight_sponge_u/v/h` are still computed elsewhere but are no longer read by
  this method (same situation the 2026-07-01 `jswm.py` entry noted); left in
  place as harmless.

### Verification done
- `python -m py_compile mapping/src/mod.py` passes.
- Normalised `diff` of the ported block against `jswm.py` shows only the intended
  rename / cosmetic differences.
- Functional smoke test (stub instance, 8×10 grid, 2 omegas × 3 thetas): the new
  `_compute_IT_2D` returns correctly-shaped finite `u (ny,nx-1)`, `v (ny-1,nx)`,
  `h (ny,nx)` for all three `bc_it_method` values (`plane_wave`,
  `plane_wave_bdy`, `wkb`) and for both `flag_tangent=True/False`.

### Not done / open questions
- `Model_bmit._compute_IT_2D` still holds the old loop implementation (out of the
  requested scope); update it the same way if that model needs the newest medium.
- `_compute_IT_2D` is jit-compiled in `__init__` but not called elsewhere in
  `mod.py` (the live model path uses `self.swm.compute_IT_2D`); no end-to-end
  model run exercises this copy, so only the stand-alone smoke test above was run.

---

## Change 2026-07-03 — Per-constituent internal-tide generation control `itg` (rename old `itg` → `itg_coeff`)

### What
Two changes to the internal-tide generation control in `Model_csw1l` / `CSWm`:
1. **Renamed** the existing `(ny,nx)` coefficient control `itg` → `itg_coeff`
   (the `generation*(u_bar*grad_H_x+v_bar*grad_H_y)*(1+itg_coeff)` forcing).
2. **Added** a new richer control `itg`, ported from
   `Bellemin-Laponnaz_2026_JAMES` (MASSH_val): a per-constituent, time-dependent
   parameter of shape `(n_omega, 4, ny, nx)` producing
   `rhs_itg = Σ_i grad_H_x*tidal_U[i]*(itg[i,0]cos w_i t + itg[i,1]sin w_i t)`
   `        + grad_H_y*tidal_V[i]*(itg[i,2]cos w_i t + itg[i,3]sin w_i t)`.
   The two forcings are **mutually exclusive** (validated in `__init__`).
Also added the `Basis_gauss_itg` reduced basis to control `itg`.

### Why
Bring MASSH_val's per-constituent internal-tide generation parametrization into
the generation codebase, while keeping the previously-added coefficient forcing
(renamed to avoid the name clash).

### Files changed
- `mapping/models/model_sw1l/jswm.py` (`CSWm`)
- `mapping/src/mod.py` (`Model_csw1l`)
- `mapping/src/basis.py`
- `mapping/src/config_default.py`

### Details
- **jswm.py**: `compute_rhs_itg` arg `itg`→`itg_coeff`; new
  `compute_rhs_itg_omega(t, itg)` (uses `grad_H_x`/`grad_H_y` — the topography
  gradient — in place of the bathymetry gradient, plus per-constituent
  `tidal_U`/`tidal_V` and `omegas`). `_step_rk4_nstep`/`_step_euler_nstep` gain
  an `itg` kwarg; because this forcing is **time-dependent** it is recomputed
  **inside** the scan body from the running time `tc` (the `itg_coeff` forcing
  stays a constant computed before the scan). New `CSWm` defaults
  `flag_itg_omega`, `tidal_U`, `tidal_V` (set by the wrapper). Also fixed the
  pre-existing missing `return` in the euler scan body.
- **mod.py**: registers `State.params['itg']` of shape `(n_omega,4,ny,nx)`;
  threads `itg` as a second differentiated control immediately after `itg_coeff`
  through `_jstep`/`_jstep_tgl`/`_jstep_adj` (signatures, closures,
  primals/tangents/cotangents, adjoint unpack/accumulate/return) and
  `step`/`step_tgl`/`step_adj`. `init_tidal_velocity` extended to build
  per-constituent `self.tidal_U`/`self.tidal_V` `(n_omega,ny,nx)` and
  `self.omega_names` from the atlas already in `config.MOD.path_tidal_model`
  (via new helper `_read_tidal_amplitude`, reusing `grid.interp2d`); no new atlas
  path. `__init__` sets the forcing fields on `self.swm` and validates
  mutual-exclusivity + prerequisites (`path_tidal_model`, non-NaN
  `tidal_U`/`tidal_V`/`dH_dx`/`dH_dy`, `len(omega_names)==len(omegas)`).
- **basis.py**: new `Basis_gauss_itg` (dispatcher super `BASIS_GAUSS_ITG`);
  Gaussian decomposition on the `ny*nx` axis, `(n_omega,4)` stacked;
  `operg`/`operg_transpose` follow this repo's JAX basis API (vjp-based reduced
  projection like `Basis_gauss2d_jax`).
- **config_default.py**: `MOD_CSW1L` `name_params` comment lists `itg_coeff`/`itg`;
  `flag_itg` comment reworded; new `name_var_tidal_amp` key; new `BASIS_GAUSS_ITG`
  block (`name_mod_var='itg'`, `facns`, `D_itg`, `sigma_Q`, `Nwaves`).

### Differences vs. MASSH_val source (important for merging)
- Uses `self.dH_dx`/`self.dH_dy` (→ `swm.grad_H_x`/`grad_H_y`) and `self.H`
  instead of `bathymetry`/`grad_bathymetry_*`; **no** `bathymetry` reading added.
- Per-constituent `tidal_U`/`tidal_V` are sourced from the existing
  `config.MOD.path_tidal_model` atlas (extending `init_tidal_velocity`), not from
  a new `path_tidal_velocity` directory as in the reference.
- `itg` is threaded as a differentiated JAX control (attribute-injection
  convention), not via the reference's `slice_params`/`one_step` architecture.

### Verification done
- `python -m py_compile` passes on all four files.
- Grep audit: every `_jstep*` signature / closure / primals / tangents /
  cotangents / call site carries BOTH `itg_coeff` and `itg`; all `'itg'` keys now
  refer to the new control, `'itg_coeff'` to the old coefficient. Confirmed the
  other consumer of `_step_*_nstep` (`Model_bmit.model_it_step_nstep`) does not
  pass `itg`, so it defaults to `None`.

### Not done / open questions
- Amplitude variable names / units in the `path_tidal_model` atlas files are
  assumed FES `Ua`/`Va` in cm/s (overridable via `config.MOD.name_var_tidal_amp`);
  verify against the real files.
- `omega_names` order (atlas dict keys) must line up with `config.MOD.w_waves`;
  a length check is enforced but not an order check.
- No end-to-end / 4DVar tangent-adjoint run performed (needs a fixture with
  `file_mode_aux`, `path_tidal_model`, and a bathymetry `H`). Recommend a forward
  run with `name_params=['itg']`, a gradient test on `itg`, and a
  `Basis_gauss_itg` `operg`/`operg_transpose` dot-product test.

---

<!--
TEMPLATE for future entries — copy below this comment for each new change.

## Change YYYY-MM-DD — <short title>

### What
### Why
### Files changed
### Details
### Verification done
### Not done / open questions
-->
