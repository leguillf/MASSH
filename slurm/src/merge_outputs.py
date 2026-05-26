#!/usr/bin/env python3
import os
import sys
import pickle
import multiprocessing as mp
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

import xarray as xr
from pathlib import Path

# Add MASSH mapping path — override with the MASSH_PATH environment variable
# Default: resolve relative to this script's location (slurm/ → mapping/)
_MASSH_PATH = os.environ.get('MASSH_PATH', str(Path(__file__).parent.parent / 'mapping'))
sys.path.append(_MASSH_PATH)
from src import exp, state
from src.run_assimilation import (
    merge_output_date,
    generate_dates,
    parallel_merge,
    merge_time_windows_outputs,
)

# -------------------- Logging helpers --------------------
def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

# -------------------- Main merge workflow --------------------
def merge_outputs(
    path_config,
    dir_save_pickle=None,
    name_var_save=['sla'],
    num_workers=4,
    plot=False,
    iw_start=0,
    iw_end=None,
    merge_time_windows=True,
    skip_spatial_merge=False,
    force=False,
    rank=0,
    world=1,
):
    log(f"Loading configuration from {path_config}")
    with open(path_config, "rb") as f:
        config = pickle.load(f)

    if dir_save_pickle is None:
        raise ValueError(
            "dir_save_pickle must be provided (root directory where prepare_VarDyn.py wrote its pickles)"
        )
    path_save_pickle = os.path.join(dir_save_pickle, config.EXP.name_experiment)

    # Load precomputed pickle files
    with open(f'{path_save_pickle}/weights.pkl', "rb") as f:
        data = pickle.load(f)
    weights_space_sum = data['weights_space_sum']
    list_tile_paths = data.get('list_tile_paths')
    # Per-tile weights/interpolators are loaded on-the-fly by merge_output_date

    with open(f'{path_save_pickle}/dates.pkl', "rb") as f:
        list_date_start, list_date_middle, list_date_end = pickle.load(f)
    with open(f'{path_save_pickle}/list_State.pkl', "rb") as f:
        list_State_all = pickle.load(f)

    kernel = Gaussian2DKernel(x_stddev=1, y_stddev=1)

    # Merge all spatial subwindows per time window
    n_windows = len(list_date_start)

    if iw_end is None or iw_end > n_windows:
        iw_end = n_windows
    
    if iw_start < 0 or iw_start >= iw_end:
        raise ValueError(f"Invalid time window range: [{iw_start}, {iw_end})")
    
    if not skip_spatial_merge:
        log(f"Spatial merge for time windows {iw_start} → {iw_end}")
        
        for iw in range(iw_start, iw_end):
            date_start  = list_date_start[iw]
            date_middle = list_date_middle[iw]
            date_end    = list_date_end[iw]
            State_window = list_State_all[iw]
            
            log(f"Processing time window {iw}: {date_start} → {date_end}")
            
            config0 = config.copy()
            config0.EXP = config0.EXP.copy()
            config0.EXP.tmp_DA_path += f'/subwindow_{str(date_middle)[:10]}'
            config0.EXP.path_save += f'/subwindow_{str(date_middle)[:10]}'
            State0 = state.State(config0)
            dates_window = generate_dates(date_start, date_end, config0.EXP.saveoutput_time_step)

            # Shard dates across array tasks
            if world > 1:
                dates_window = dates_window[rank::world]
                log(f"  rank {rank}/{world}: {len(dates_window)} dates assigned")

            if not force:
                expected_outputs = [
                    os.path.join(
                        config0.EXP.path_save,
                        f'{config0.EXP.name_experiment}'
                        f'_y{date.year}'
                        f'm{str(date.month).zfill(2)}'
                        f'd{str(date.day).zfill(2)}'
                        f'h{str(date.hour).zfill(2)}'
                        f'm{str(date.minute).zfill(2)}.nc'
                    )
                    for date in dates_window
                ]
                if expected_outputs and all(os.path.exists(path) for path in expected_outputs):
                    log(f"Skipping time window {iw}: merged outputs already exist")
                    continue

            parallel_merge(dates_window, State0, State_window, name_var_save, kernel, None, weights_space_sum, None, list_tile_paths=list_tile_paths, num_workers=num_workers)
    else:
        log("Skipping spatial merge (already done)")

    # Merge time windows
    time_overlap = (list_date_end[0] - list_date_start[1]).days if len(list_date_start) > 1 else 0
    if merge_time_windows:
        log("Merging overlapping time windows")
        merge_time_windows_outputs(
            config,
            list_date_start,
            list_date_middle,
            list_date_end,
            time_overlap,
        )
    else:
        log("Skipping final time-window merge")

# -------------------- Script entry point --------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge DA outputs for an experiment.")
    parser.add_argument("path_config", type=str, help="Path to experiment config pickle")
    parser.add_argument("--dir_save_pickle", type=str, required=True,
                        help="Root directory where prepare_VarDyn.py wrote its pickles (DIR_SAVE_PICKLE)")
    parser.add_argument(
        "--name_var_save",
        type=str,
        default='sla',
        help="Comma-separated list of variables to merge, e.g. sla,ssh,ugos,vgos"
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers per time window")

    parser.add_argument(
        "--iw_start",
        type=int,
        default=0,
        help="Start index of time windows to merge (inclusive)"
    )
    parser.add_argument(
        "--iw_end",
        type=int,
        default=None,
        help="End index of time windows to merge (exclusive)"
    )

    parser.add_argument(
        "--merge_time_windows",
        action="store_true",
        help="Merge overlapping time windows (final step)"
    )
    parser.add_argument(
        "--skip_spatial_merge",
        action="store_true",
        help="Skip spatial merge (use when spatial merges are already done)"
    )
    parser.add_argument("--force", action="store_true", help="Force recomputing merged outputs even if they already exist")
    parser.add_argument("--rank", type=int, default=0, help="Rank of this task (for date sharding across SLURM array tasks)")
    parser.add_argument("--world", type=int, default=1, help="Total number of tasks sharing the merge")

    args = parser.parse_args()

    name_var_save = [v.strip() for v in args.name_var_save.split(",")]

    log(f"Starting merge with {args.num_workers} workers")
    merge_outputs(
        args.path_config,
        dir_save_pickle=args.dir_save_pickle,
        name_var_save=name_var_save,
        num_workers=args.num_workers,
        iw_start=args.iw_start,
        iw_end=args.iw_end,
        merge_time_windows=args.merge_time_windows,
        skip_spatial_merge=args.skip_spatial_merge,
        force=args.force,
        rank=args.rank,
        world=args.world,
    )
    log("Merge finished successfully")
