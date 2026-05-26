# VarDyn SLURM

Scripts to run [VarDyn/MASSH](https://github.com/leguillf/MASSH) SSH mapping experiments on HPC clusters using SLURM GPU arrays. Lives in `slurm/` inside the MASSH repository.

## Overview

Large-scale SSH mapping with MASSH (e.g. global VarDyn runs) is parallelised over:
- **Space** — the domain is split into overlapping spatial tiles
- **Time** — the time period is split into overlapping time windows

Each SLURM array task (one GPU) processes a dynamic subset of tiles, then participates in a distributed spatial merge. A final task-0-only step merges all time windows into the full output.

```
sbatch slurm/run/VarDyn_GLO.sh [--skip-prepare] [--restart] [--force-merge] [--name_exp <name>]
```

## Repository structure

```
slurm/
├── run/
│   └── VarDyn_GLO.sh       # Example SLURM array job script (copy & edit per experiment)
└── src/
    ├── prepare_VarDyn.py   # Prepare spatial/temporal subwindows and save pickles
    ├── run_tile.py         # Run one assimilation tile (called per-GPU in parallel)
    └── merge_outputs.py    # Merge spatial tiles and time windows into final output
```

> Config files (`.py`) live in a sibling `configs/` directory and are not part of this repo.

## Workflow

```
Task 0                              Tasks 1…N-1
──────────────────────────────────  ──────────────────────────
prepare_VarDyn.py  ──── barrier ──► wait for "prepared" file
        │
        ▼
  for each time window:
    write tile list  ──── signal ──► read tile list
                                     claim & run tiles (atomic mkdir)
    barrier tw{i}   ◄──────────────  barrier tw{i}
    merge (rank 0)                   merge (rank k/N)
    barrier merge{i} ◄─────────────  barrier merge{i}
        │
        ▼
  merge_time_windows (task 0 only)
```

## Scripts

### `VarDyn_GLO.sh`

Example SLURM submission script — copy and edit the **USER SETTINGS** block for each experiment.

| Variable | Description |
|---|---|
| `MASH_DIR` | **Absolute path to the MASSH repo root** — required because SLURM copies the script to a spool directory before execution, making `$0`-relative paths unreliable |
| `NUM_GPUS` | Number of GPU array tasks (also update `#SBATCH --array`) |
| `DIR_SAVE_PICKLE` | Root directory for all pickle/output files |
| `PATH_CONFIG` | Path to the main MASSH config `.py` |
| `PATH_CONFIG_EQ` | Path to the equatorial MASSH config `.py` |
| `INIT_DATE` / `FINAL_DATE` | Experiment date range |
| `NAME_VAR` | Comma-separated list of variables to save |
| `GRID_TYPE` / `NX_PROC` … | Spatial subwindow grid parameters |
| `SPACE_WIN_X/Y`, `SPACE_OVERLAP_X/Y` | Spatial window size and overlap (degrees) |
| `TIME_WIN`, `TIME_OVERLAP` | Temporal window size and overlap (days) |
| `FLAG_INIT` / `FLAG_BACKGROUND` / `NAME_EXP` | Initialise from / use background from a previous experiment |
| `BARRIER_TIMEOUT` | Seconds to wait at each inter-GPU barrier before proceeding (default: 14400 = 4 h) |

**CLI flags** (passed after the script name):

| Flag | Effect |
|---|---|
| `--skip-prepare` | Skip `prepare_VarDyn.py` if pickles already exist |
| `--restart` | Pass `--restart` to `run_tile.py` (resume from checkpoint) |
| `--force-merge` | Force re-merge even if output files already exist |
| `--merge-only` | Skip preparation and assimilation, only run spatial and time-window merges |
| `--name_exp <name>` | Override experiment name (default: read from config or filename) |

**`EXP_NAME` resolution order:**
1. `--name_exp` CLI flag
2. `name_experiment = '...'` variable in `PATH_CONFIG`
3. Config filename with `config_` prefix stripped

**Barrier robustness** (Lustre/GPFS):
- `mkdir -p` for the barrier directory is retried up to 5 times with backoff
- `touch` inside `barrier_wait` is similarly retried
- A 30-minute timeout (`BARRIER_TIMEOUT`) prevents a dead task from blocking the whole job

### `prepare_VarDyn.py`

Reads the MASSH config files and generates the pickle tree under `DIR_SAVE_PICKLE/<EXP_NAME>/`:
```
<EXP_NAME>/
  config.pkl
  subwindow_<date>/
    subwindow_<space>/
      config.pkl
      state.pkl
      weights.pkl
```

### `run_tile.py`

Loads one `subwindow_<space>` pickle directory and runs the full MASSH assimilation (forward + inverse). Writes `Xres.nc` on completion.

### `merge_outputs.py`

Two-stage merge:
1. **Spatial merge** (parallelised over dates, distributed over ranks): blends overlapping tiles with Gaussian-tapered weights.
2. **Time-window merge** (task 0 only): concatenates spatial merges across all time windows.

## Usage example

```bash
# Submit with 6 GPUs (array 0-5 in the script)
sbatch VarDyn_GLO.sh

# Skip re-preparation if pickles already exist
sbatch VarDyn_GLO.sh --skip-prepare

# Resume a crashed run
sbatch VarDyn_GLO.sh --skip-prepare --restart

# Override experiment name
sbatch VarDyn_GLO.sh --name_exp my_custom_name
```

## Requirements

- SLURM with GPU support (`--gpus=v100_32g:1` or similar)
- MASSH repository — `mapping/` is located automatically relative to `slurm/` (no `MASSH_PATH` env var needed when running from within the repo)
- Python environment with: `numpy`, `xarray`, `scipy`, `astropy`, `jax`, `cartopy`
- Lustre/GPFS shared filesystem (barrier mechanism uses atomic `mkdir`)

## Notes

- Tile claiming uses `mkdir` (atomic on all POSIX filesystems including Lustre/GPFS) — no NFS locking required.
- Set `HDF5_USE_FILE_LOCKING=FALSE` if you encounter NetCDF read errors on shared filesystems (already handled inside `run_assimilation.py`).
- Logs are written to `./logs/<EXP_NAME>_job-<JOB_ID>/gpu<ARRAY_ID>.log` and per-tile under subdirectories.
