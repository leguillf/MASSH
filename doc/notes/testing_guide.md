# MASSH testing guide

This guide describes standard validation commands for day-to-day development.
Use the smallest scope that gives confidence, then escalate to the full suite.

## 1) Environment setup

```bash
source ~/.bashrc >/dev/null 2>&1 || true
conda activate massh
cd /home/flo/MASSH
```

Some modules import from `mapping/src` using `from src...`.
If you run scripts from the repo root, set:

```bash
export PYTHONPATH=/home/flo/MASSH/mapping
```

## 2) Fast sanity checks

### Python syntax checks

```bash
python -m py_compile mapping/src/mod.py
python -m py_compile mapping/models/model_qgsw/sw.py
python -m py_compile mapping/models/model_qg1l/jqgm.py
```

Notes:
- Run the first two whenever you edit those files.
- Add more files to `py_compile` if your changes are localized elsewhere.

### Focused smoke test (QG model)

```bash
PYTHONPATH=/home/flo/MASSH/mapping python - <<'PY'
import numpy as np
import jax.numpy as jnp
from mapping.models.model_qg1l.jqgm import Qgm

ny, nx = 10, 12
dx = jnp.ones((ny, nx))*10000.
dy = jnp.ones((ny, nx))*10000.
f = jnp.ones((ny, nx))*1e-4
c = jnp.ones((ny, nx))*2.7
ssh = jnp.zeros((ny, nx))

m = Qgm(dx=dx, dy=dy, dt=600., SSH=np.array(ssh), c=np.array(c), f=np.array(f),
        formulation='ssh', advect_pv=True, ageo_velocities=False, time_scheme='rk3')
out = m.step(jnp.zeros((1, ny, nx)), jnp.zeros((1, ny, nx)), nstep=2)
print('stacked_shape', out.shape)

m2 = Qgm(dx=dx, dy=dy, dt=600., SSH=np.array(ssh), c=np.array(c), f=np.array(f),
         formulation='sf', advect_pv=True, ageo_velocities=False, time_scheme='rk2')
out2 = m2.step(jnp.zeros((ny,nx)), jnp.zeros((ny,nx)), nstep=2)
print('ssh_shape', out2.shape)
PY
```

Expected output:
- `stacked_shape (1, 10, 12)`
- `ssh_shape (10, 12)`

## 3) Run automated tests

### Full pytest suite

```bash
cd /home/flo/MASSH
pytest -q tests
```

### Run a single test module

```bash
pytest -q tests/test_qg_comparison.py
pytest -q tests/test_calibrate_bmaux_synthetic.py
```

### Run one specific test

```bash
pytest -q tests/test_qg_comparison.py -k "name_fragment"
```

## 4) Suggested validation ladder

1. `py_compile` on touched files.
2. One focused smoke test for the modified subsystem.
3. One or more targeted pytest modules.
4. Full `pytest -q tests` before merging substantial refactors.

## 5) Troubleshooting

- `ModuleNotFoundError: src`: set `PYTHONPATH=/home/flo/MASSH/mapping`.
- `ModuleNotFoundError: jax`: activate `massh` env and ensure dependencies from `environment.yml` are installed.
- If tests rely on heavy external data, start with synthetic/unit tests in `tests/` first.
