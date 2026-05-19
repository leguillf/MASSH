# 12. Typical end-to-end flow

```python
from src import exp, state, mod, bc, obs, obsop, basis, inv, diag

config   = exp.Exp('config_my_run.py')
State    = state.State(config)
Model    = mod.Model(config, State)
Bc       = bc.Bc(config, State)
dict_obs = obs.Obs(config, State)
Obsop    = obsop.Obsop(config, State, dict_obs, Model)
Obsop.process_obs()
Basis    = basis.Basis(config, State)
Basis.set_basis(...)                # builds Q, locations, hyperparams

inv.Inv_4Dvar(config=config, State=State, Model=Model,
              dict_obs=dict_obs, Obsop=Obsop, Basis=Basis, Bc=Bc)

diag.Diag(config, State)
```

For tiled / large-domain runs the `run_assimilation.prepare_process(...)`
helper takes the same building blocks and dispatches them across windows.
