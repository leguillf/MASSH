# 6. Observations (`OBS`) and observation operator (`OBSOP`)

`Obs(config, State)` ([obs.py](../../mapping/src/obs.py)) returns `dict_obs`, a
dictionary keyed by observation timestamp; values are dicts holding the file
paths, source name, variable name, and metadata required to assimilate that
batch. Three super-blocks exist:

- `OBS_SSH_NADIR` — 1D nadir-altimeter tracks
- `OBS_SSH_SWATH` — 2D SWOT-like swaths
- `OBS_L4` — gridded L4 (used as ground truth in OSSE diagnostics or as a
  validation reference)

`Obsop(config, State, dict_obs, Model)` ([obsop.py](../../mapping/src/obsop.py))
builds the observation operator H. It precomputes per-timestamp interpolation
weights and caches them on disk (the cache key contains the joined `name_obs`
list, so disjoint subsets do not collide). Public methods used elsewhere:

- `Obsop.process_obs()` — finalize cache
- `Obsop.is_obs_time(t)` — fast membership test
- `Obsop.misfit(t, State)` — `(y - H x) / σ` evaluated at time `t`

`OBSOP_INTERP_L3*` interpolates the model state onto sparse along-track /
swath observations; `OBSOP_INTERP_L4` interpolates onto a regular L4 grid.
