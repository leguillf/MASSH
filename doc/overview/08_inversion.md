# 8. Inversion (`INV`)

[inv.py](../../mapping/src/inv.py) defines:

- `Inv_forward(config, State, Model, Bc)` — pure forward integration with
  output saving (used when `NAME_INV` is `None`).
- `Inv_4Dvar(config, State, Model, dict_obs, Obsop, Basis, Bc)` —
  incremental 4DVar in the reduced basis.

## 4DVar in MASSH

The control vector `X` lives in basis space. The cost function is

$$J(X) = \tfrac12 X^\top Q^{-1} X + \tfrac12 \sum_t \big\| (y_t - H_t M_{0\to t}(\Phi(X))) / \sigma_t \big\|^2$$

where $\Phi$ = `Basis.operg`, $M$ = `Model.step`, $H$ = `Obsop`, and $Q$ is the
diagonal prior carried by the basis. Gradients are obtained by combining the
basis adjoint, JAX `vjp` of the JAX-models, and the obsop adjoint. The outer
optimizer is `scipy.optimize.minimize` with method `L-BFGS-B`, controlled by:

- `INV.maxiter` — outer iteration cap
- `INV.ftol`, `INV.gtol` — stopping tolerances
- `INV.timestep_checkpoint` — frequency at which `Basis.operg` is reapplied
  along the trajectory
- `INV.path_init_4Dvar` — optional warm start (`Xres.nc` from a prior run)
- `INV.path_save_control_vectors` — destination for `Xres.nc` (final control
  vector, used by downstream diagnostics and reruns).

The custom exceptions `ConvergenceReached` / `CrazyGradient` short-circuit the
optimizer when `ftol`/`gtol` are met or when a NaN/Inf gradient is detected.
