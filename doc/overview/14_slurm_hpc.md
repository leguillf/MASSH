# 14. SLURM HPC execution

The `slurm/` directory contains scripts for running large-scale VarDyn/MASSH
experiments on HPC clusters via SLURM GPU arrays.  A single `sbatch` command
fans out the workload over many GPUs, synchronises via filesystem barriers,
and merges the results back into a single output file.

## Parallelisation strategy

Large runs are parallelised over two dimensions:

- **Space** — the domain is split into overlapping spatial tiles.
- **Time** — the time period is split into overlapping time windows.

Each SLURM array task (one GPU) processes a dynamic subset of tiles for the
current time window, then participates in a distributed spatial merge.  A
final task-0-only step merges all time windows into the full output.

## Files

| File | Role |
|------|------|
| `VarDyn_GLO.sh` | Example SLURM array job script — copy & edit the **USER SETTINGS** block per experiment |
| `prepare_VarDyn.py` | Reads MASSH config(s) and writes the pickle tree under `DIR_SAVE_PICKLE/<EXP_NAME>/` |
| `run_tile.py` | Loads one `subwindow_<space>` pickle dir and runs the full MASSH assimilation; writes `Xres.nc` |
| `merge_outputs.py` | Two-stage merge: (1) spatial — Gaussian-tapered blend of overlapping tiles, distributed over ranks; (2) time-window — concat across all windows (task 0 only) |

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

Tile claiming uses `mkdir` (atomic on all POSIX filesystems including
Lustre/GPFS) — no NFS locking is required.

## Submission

```bash
sbatch VarDyn_GLO.sh [--skip-prepare] [--restart] [--force-merge] \
                     [--merge-only] [--name_exp <name>]
```

| Flag | Effect |
|------|--------|
| `--skip-prepare` | Skip `prepare_VarDyn.py` if pickles already exist |
| `--restart` | Pass `--restart` to `run_tile.py` (resume from checkpoint) |
| `--force-merge` | Force re-merge even if output files already exist |
| `--merge-only` | Skip preparation and assimilation, only run merges |
| `--name_exp <name>` | Override experiment name |

**`EXP_NAME` resolution order:**
1. `--name_exp` CLI flag
2. `name_experiment = '...'` variable in `PATH_CONFIG`
3. Config filename with `config_` prefix stripped

## Key settings in `VarDyn_GLO.sh`

| Variable | Description |
|----------|-------------|
| `NUM_GPUS` | Number of GPU array tasks (also update `#SBATCH --array`) |
| `DIR_SAVE_PICKLE` | Root directory for all pickle/output files |
| `PATH_CONFIG` / `PATH_CONFIG_EQ` | Paths to the main and equatorial MASSH config `.py` |
| `INIT_DATE` / `FINAL_DATE` | Experiment date range |
| `NAME_VAR` | Comma-separated list of variables to save |
| `SPACE_WIN_X/Y`, `SPACE_OVERLAP_X/Y` | Spatial window size and overlap (degrees) |
| `TIME_WIN`, `TIME_OVERLAP` | Temporal window size and overlap (days) |
| `FLAG_INIT` / `FLAG_BACKGROUND` / `NAME_EXP` | Seed from a previous experiment |
| `BARRIER_TIMEOUT` | Seconds to wait at each inter-GPU barrier (default: 14400 = 4 h) |

## Barrier robustness

Barriers use filesystem `mkdir`; on Lustre/GPFS the `mkdir` and `touch`
operations are retried up to 5 times with exponential backoff.  A
`BARRIER_TIMEOUT` (default 4 h) prevents a dead task from blocking the whole
job.

## Logs

Written to `./logs/<EXP_NAME>_job-<JOB_ID>/gpu<ARRAY_ID>.log` and per-tile
under subdirectories.

## Requirements

- SLURM with GPU support (`--gpus=v100_32g:1` or similar)
- Shared filesystem (Lustre/GPFS recommended)
- Python environment: `numpy`, `xarray`, `scipy`, `astropy`, `jax`, `cartopy`
- Set `HDF5_USE_FILE_LOCKING=FALSE` on shared filesystems if NetCDF read
  errors occur (already handled inside `run_tile.py`).
