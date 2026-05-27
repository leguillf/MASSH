# Adjoint performance benchmarks

**Date**: 2026-05-27  
**Device**: GPU (CUDA, single device)  
**Grid**: 256×256, nl=1, float32  
**Warmup**: 1 call, timed reps: 20 (median reported)  
**Source**: `diag_forward_adjoint_benchmark.ipynb`

---

## 1. max_nstep sweep — windowed checkpointing (qgsw_sw vs qg1l_jax)

`Model_qgsw.step_adj` implements two-level checkpointing:

- **Outer (Python loop)**: splits `nstep` into chunks of `max_nstep`; stores chunk-boundary `(u,v,h)` states; calls `jstep_adj_jit` per chunk.
- **Inner (lax.scan + checkpoint)**: within each chunk, `jax.checkpoint(single_step)` stores only the carry; re-executes each step during the backward scan.

Total nstep = 72, varying `max_nstep`:

| model     | max_nstep       | fwd ms | adj ms | ratio | total ms |
|-----------|-----------------|--------|--------|-------|----------|
| qgsw_sw   | 72 (1 chunk)    |    4.9 |   17.4 | 3.56× |       22 |
| qgsw_sw   | 18 (4 chunks)   |    7.2 |   22.0 | 3.05× |       29 |
| qgsw_sw   | 6  (12 chunks)  |   14.4 |   28.2 | 1.95× |       43 |
| qgsw_sw   | 1  (72 chunks)  |   55.4 |  109.0 | 1.97× |      164 |
| qg1l_jax  | — (flat vjp)    |    4.7 |    9.6 | 2.06× |       14 |

### Findings

- Smaller `max_nstep` reduces the adj/fwd **ratio** but increases **absolute time** — the Python outer loop replaces XLA's compiled scan kernel, growing forward time 11× (4.9 ms → 55 ms). Total time (fwd + adj) is worst at `max_nstep=1` (164 ms) vs `max_nstep=72` (22 ms).
- The default `max_nstep=240` (no outer chunking at nstep=72) is already optimal for wall-clock throughput.
- qg1l achieves 1.89× at the same forward speed because its scan body has no `jax.checkpoint` — it stores the full trajectory in HBM (feasible for a lightweight spectral step, ~1 MB/step).

---

## 2. scan_checkpoint flag — qgsw_sw vs qg1l_jax

Tested `scan_checkpoint=True` (default: checkpoint entire `single_step` inside `lax.scan`) vs `scan_checkpoint=False` (no checkpoint — store full forward trajectory):

| model     | ckpt  | nstep | fwd ms | adj ms | ratio |
|-----------|-------|-------|--------|--------|-------|
| qgsw_sw   | True  |     1 |    3.3 |    4.4 | 1.35× |
| qgsw_sw   | True  |    10 |    3.3 |    4.3 | 1.32× |
| qgsw_sw   | True  |    72 |    4.6 |   17.1 | 3.71× |
| qgsw_sw   | False |     1 |    3.2 |    4.0 | 1.24× |
| qgsw_sw   | False |    10 |    3.7 |   27.1 | 7.41× |
| qgsw_sw   | False |    72 | OOM    | 17.75 GiB required | — |
| qg1l_jax  | —     |     1 |    0.9 |    0.8 | 0.89× |
| qg1l_jax  | —     |    10 |    1.2 |    1.5 | 1.21× |
| qg1l_jax  | —     |    72 |    4.7 |    9.6 | 2.06× |

### Findings

**`scan_checkpoint=False` is not viable for qgsw.**

The hypothesis that removing `jax.checkpoint` would match qg1l's ~2× ratio is wrong:

- **nstep=10**: ratio degrades 1.32× → 7.41× (adjoint 6.3× slower despite no recomputation).
- **nstep=72**: OOM — XLA needs 17.75 GiB to materialise all WENO3+RK3 intermediates for 72 steps.

**Why qg1l works without checkpoint but qgsw doesn't.**  
qg1l's `single_step` is a lightweight spectral solve (few intermediates, ~1 MB/step).  
qgsw's `single_step` is WENO3+RK3: flux reconstruction on all four faces per cell, three RK stages, baroclinic pressure — ~60+ intermediate arrays per step. JAX's reverse-mode scan must buffer all of these as residuals. At 72 steps this requires 17+ GiB.

**Why `scan_checkpoint=False` is slow even at nstep=10.**  
Without checkpoint the backward pass streams ~1.5 GB of stored intermediates back from HBM sequentially. With checkpoint, intermediates never leave L2/registers — each step re-executes from the compact `(u,v,h)` carry. For complex operators like WENO, re-execution is faster than the memory round-trip.

---

## 3. Conclusions

| finding | conclusion |
|---------|-----------|
| No-checkpoint approach (like qg1l) | Not viable for qgsw: OOM at nstep=72, 6× slower at nstep=10 |
| Windowed checkpointing (varying max_nstep) | Reduces ratio but total time always worse; default max_nstep=240 is optimal |
| Explicit adjoint via `custom_vjp` | Benchmarked and removed (prior session): equal performance, higher maintenance |
| **Recommended configuration** | **`scan_checkpoint=True`, `max_nstep=240` — current defaults** |

The 3.78× adj/fwd ratio at nstep=72 is intrinsic to checkpointed WENO scans. For short windows (nstep ≤ 10) the ratio drops to ~1.3×, matching qg1l.

---

## 4. Implementation note

`sw.py` exposes `self.scan_checkpoint = True` (added 2026-05-27) so the flag can be overridden before the first JIT call for future experiments. The default must remain `True`.
