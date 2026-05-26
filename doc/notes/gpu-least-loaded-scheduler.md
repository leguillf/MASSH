# GPU least-loaded scheduler for multiprocessing assimilation

**Date:** 2026-05-25  
**File:** `mapping/src/run_assimilation.py`  
**Notebook:** `run_VarDyn-SW_eNATL60-BLB002_GulfStream_multiwindows.ipynb`

---

## Problem

`prepare_process` assigned GPUs to assimilation subprocesses via **round-robin at creation time** — the GPU was baked into each `mp.Process` via `partial(inv.Inv_4Dvar, ..., gpu_device=gpu_devices[id_gpu])`.  
`run_assimilation_time_window` started processes up to a global `nprocs` cap but was **blind to per-GPU load**: a short job could free a slot on GPU 1 while all new processes were still queued for GPU 0.

## Solution

Moved GPU assignment from **creation time** to **start time** using a least-loaded counter.

### `prepare_process`

- Removed `id_gpu` round-robin counter entirely.
- `list_processes` now stores **callables** — `partial(inv.Inv_4Dvar, config=..., State=..., verbose=0)` — with no GPU pre-bound.

### `run_assimilation_time_window`

- Added `gpu_devices=None` optional parameter (default falls back to `['0']`).
- Added `gpu_load = {gpu_id: int}` counter.
- `active_processes` changed from `set[Process]` → `set[(Process, gpu_id)]`.
- At each start slot: pick `min(gpu_load, key=gpu_load.get)`, create a fresh `mp.Process(target=worker, kwargs={'gpu_device': gpu_id})`, increment counter.
- On reap: decrement the finished process's GPU counter.
- Global `nprocs` cap is **unchanged** — concurrency ceiling and GPU selection are orthogonal.

### Notebook

- Added `gpu_devices=gpu_devices` to the `run_assimilation_time_window` call.

---

## Design decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Where to assign GPU | Start time (not creation time) | Only at start time is the current load known |
| Load metric | Active process count per GPU | Zero external deps; fits the existing polling loop |
| Concurrency control | Keep global `nprocs` cap | Separate concern from GPU selection |
| `run_assimilation_time_window` signature | Add `gpu_devices=None` | Backward compatible; `None` → single GPU `'0'` |

---

## Related

- [[run_assimilation]] — windowed tiled assimilation orchestration
- [[inv]] — `Inv_4Dvar` sets `CUDA_VISIBLE_DEVICES` from `gpu_device` arg
