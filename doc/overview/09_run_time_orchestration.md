# 9. Run-time orchestration

[run_assimilation.py](../../mapping/src/run_assimilation.py) implements **windowed
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
