#!/usr/bin/env python3
import argparse
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import os
from pathlib import Path
from datetime import datetime
import pickle

import warnings
from pathlib import Path

# MASSH imports — override path with the MASSH_PATH environment variable
# Default: resolve relative to this script's location (slurm/ → mapping/)
_MASSH_PATH = os.environ.get('MASSH_PATH', str(Path(__file__).parent.parent / 'mapping'))
sys.path.append(_MASSH_PATH)
from src import inv


def run_tile(tile_dir: Path, restart:bool):
    """
    Run one data assimilation tile located in:
    subwindow_<time>/subwindow_<space>
    """

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "CPU-only")

    print(f"[{datetime.now()}] Starting tile")
    print(f"Using GPU(s): {gpu}")
    print(f"Tile directory: {tile_dir}")

    # --------------------------------------------------
    # Expected input files
    # --------------------------------------------------
    config_path = tile_dir / "config.pkl"
    state_path  = tile_dir / "state.pkl"

    if not config_path.exists():
        raise RuntimeError(f"Missing config file: {config_path}")
    if not state_path.exists():
        raise RuntimeError(f"Missing state file: {state_path}")

    print(f"Using config: {config_path.name}")
    print(f"Using state : {state_path.name}")

    # --------------------------------------------------
    # Load inputs
    # --------------------------------------------------
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    with open(state_path, "rb") as f:
        State = pickle.load(f)

    print(f"Running inversion, output path: {config.EXP.path_save}")

    # --------------------------------------------------
    # Run algorithm
    # --------------------------------------------------
    if restart or not os.path.exists(f'{config.INV.path_save_control_vectors}/Xres.nc'):
        inv.Inv_4Dvar(config=config, State=State, verbose=0)
        print(f"[{datetime.now()}] Finished tile: {tile_dir}")
    else:
        print(f"[{datetime.now()}] Non-processed tile: {tile_dir}")
        print(f'Because you did not ask for restart and {f'{config.INV.path_save_control_vectors}/Xres.nc'} exists.')


def main():

    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser(
        description="Run one data assimilation spatial tile"
    )
    parser.add_argument(
        "tile_path",
        type=Path,
        help="Path to subwindow_<time>/subwindow_<space>"
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart assimilation if set"
    )

    args = parser.parse_args()
    tile_dir = args.tile_path.resolve()
    RESTART = args.restart

    if not tile_dir.exists():
        print(f"ERROR: tile directory does not exist: {tile_dir}", file=sys.stderr)
        sys.exit(1)

    if not tile_dir.is_dir():
        print(f"ERROR: tile_path is not a directory: {tile_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        run_tile(tile_dir, restart=RESTART)
    except Exception as e:
        print(f"[ERROR] Tile failed: {tile_dir}", file=sys.stderr)
        print(str(e), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
