#!/usr/bin/env python3
"""
Prepare VarDyn subwindows and save pickles for HPC processing.

Usage:
    python prepare_VarDyn.py <path_config> <path_config_eq> [options]

This creates the pickle files (config, states, weights, dates) needed
by VarDyn_GLO.sh and merge_outputs.py.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add MASSH mapping path — override with the MASSH_PATH environment variable
# Default: resolve relative to this script's location (slurm/ → mapping/)
_MASSH_PATH = os.environ.get('MASSH_PATH', str(Path(__file__).parent.parent / 'mapping'))
sys.path.append(_MASSH_PATH)
from src import exp, state
from src.run_assimilation import prepare_process

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare VarDyn subwindows and save pickles for HPC processing."
    )

    # Required
    parser.add_argument("path_config", type=str, help="Path to main config file (.py)")
    parser.add_argument("path_config_eq", type=str, help="Path to equatorial config file (.py)")

    # Dates
    parser.add_argument("--init_date", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--final_date", type=str, required=True, help="End date YYYY-MM-DD")

    # Pickle output
    parser.add_argument("--dir_save_pickle", type=str, required=True, help="Directory to save pickle files")

    # Grid
    parser.add_argument("--grid_type", type=str, default="GRID_CAR")
    parser.add_argument("--grid_type_eq", type=str, default=None)
    parser.add_argument("--nx_proc", type=int, default=512)
    parser.add_argument("--ny_proc", type=int, default=128)
    parser.add_argument("--nx_proc_eq", type=int, default=512)
    parser.add_argument("--ny_proc_eq", type=int, default=256)
    parser.add_argument("--dx", type=float, default=10)
    parser.add_argument("--dy", type=float, default=10)
    parser.add_argument("--dlon", type=float, default=None,
                        help="Longitude spacing in degrees (GRID_GEO only). If None, read from config.")
    parser.add_argument("--dlat", type=float, default=None,
                        help="Latitude spacing in degrees (GRID_GEO only). If None, read from config.")

    # Space windows
    parser.add_argument("--space_window_size_proc_x", type=float, default=50)
    parser.add_argument("--space_window_size_proc_y", type=float, default=12.5)
    parser.add_argument("--space_window_size_proc_x_eq", type=float, default=50)
    parser.add_argument("--space_window_size_proc_y_eq", type=float, default=25)

    # Overlaps
    parser.add_argument("--space_overlap_x", type=float, default=2.5)
    parser.add_argument("--space_overlap_y", type=float, default=2.5)

    # Time windows
    parser.add_argument("--time_window_size_proc", type=float, default=50)
    parser.add_argument("--time_overlap", type=float, default=10)

    # Flags
    parser.add_argument("--flag_init_from_previous", action="store_true", default=True)
    parser.add_argument("--no_flag_init_from_previous", action="store_false", dest="flag_init_from_previous")
    parser.add_argument("--flag_assim", action="store_true", default=False)
    parser.add_argument("--flag_assim_restart", action="store_true", default=False)
    parser.add_argument("--flag_init", action="store_true", default=False)
    parser.add_argument("--flag_background", action="store_true", default=False)
    parser.add_argument("--name_exp", type=str, default=None)

    # Workers
    parser.add_argument("--obs_max_workers", type=int, default=1)

    # GPU (only used for round-robin assignment in pickle metadata)
    parser.add_argument("--gpu_devices", type=str, default="0,1", help="Comma-separated GPU device IDs")

    args = parser.parse_args()

    init_date = datetime.strptime(args.init_date, "%Y-%m-%d")
    final_date = datetime.strptime(args.final_date, "%Y-%m-%d")
    gpu_devices = [g.strip() for g in args.gpu_devices.split(",")]

    print(f"Loading config: {args.path_config}")
    config = exp.Exp(args.path_config)
    print(f"Loading equatorial config: {args.path_config_eq}")
    config_eq = exp.Exp(args.path_config_eq)

    print(f"Creating global state")
    State = state.State(config)

    print(f"Preparing subwindows ({init_date.date()} → {final_date.date()})")
    print(f"Pickle output: {args.dir_save_pickle}")

    prepare_process(
        config, config_eq, State,
        init_date, final_date,
        grid_type=args.grid_type,
        grid_type_eq=args.grid_type_eq,
        nx_proc=args.nx_proc, ny_proc=args.ny_proc,
        dx=args.dx, dy=args.dy,
        dlon=args.dlon, dlat=args.dlat,
        nx_proc_eq=args.nx_proc_eq, ny_proc_eq=args.ny_proc_eq,
        time_window_size_proc=args.time_window_size_proc,
        space_window_size_proc_x=args.space_window_size_proc_x,
        space_window_size_proc_y=args.space_window_size_proc_y,
        space_window_size_proc_x_eq=args.space_window_size_proc_x_eq,
        space_window_size_proc_y_eq=args.space_window_size_proc_y_eq,
        time_overlap=args.time_overlap,
        space_overlap_x=args.space_overlap_x,
        space_overlap_y=args.space_overlap_y,
        flag_init_from_previous=args.flag_init_from_previous,
        flag_init=args.flag_init,
        flag_background=args.flag_background,
        flag_assim=args.flag_assim,
        flag_assim_restart=args.flag_assim_restart,
        name_exp_init=args.name_exp,
        name_exp_background=args.name_exp,
        gpu_devices=gpu_devices,
        obs_max_workers=args.obs_max_workers,
        dir_save_pickle=args.dir_save_pickle,
    )

    print("Preparation complete.")


if __name__ == "__main__":
    main()
