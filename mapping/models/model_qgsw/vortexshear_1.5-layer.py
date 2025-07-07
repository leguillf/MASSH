import numpy as np
import os, sys
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from helmholtz import compute_laplace_dstI, dstI2D
from qg import QG
from sw import SW

from jax import numpy as jnp

import matplotlib
import matplotlib.pyplot as plt

dtype = np.float64

def grad_perp(f, dx, dy):
    """Orthogonal gradient"""
    return (f[..., :-1] - f[..., 1:]) / dy, (f[..., 1:, :] - f[..., :-1, :]) / dx


## Space discretization definition -------------------------------------------
nx = 192
ny = 192
nl = 1
L = 100000  # 100km
Lx = L
Ly = L
dx = np.array(Lx / nx, dtype=dtype)
dy = np.array(Ly / ny, dtype=dtype)
x_, y_ = (
    np.linspace(-Lx / 2, Lx / 2, nx + 1),
    np.linspace(-Ly / 2, Ly / 2, ny + 1),
)
x, y = np.meshgrid(x_, y_, indexing="ij")
## ---------------------------------------------------------------------------

## Layers Reference thickness definition -------------------------------------
# H = torch.zeros(nl, 1, 1, dtype=dtype, device=device)
# if nl == 1:
#     H[0, 0, 0] = 1000  # 1km
# 1-layer reduced gravity
H = np.array([[[400]]], dtype=dtype)
## ---------------------------------------------------------------------------

## Reduced Gravity definition ------------------------------------------------
# g_prime = torch.zeros(nl, 1, 1, dtype=dtype, device=device)
# if nl == 1:
#     g_prime[0, 0, 0] = 10.0
# 1-layer reduced gravity
g_prime = np.array([[[0.05]]], dtype=dtype)
## ---------------------------------------------------------------------------


## Circular Mask definition --------------------------------------------------
xc_ = 0.5 * (x_[1:] + x_[:-1])
yc_ = 0.5 * (y_[1:] + y_[:-1])
xc, yc = np.meshgrid(xc_, yc_, indexing="ij")
rc =  np.sqrt(xc**2 + yc**2)
# circular domain mask
apply_mask = False
mask = (rc < L / 2).astype(np.float64) if apply_mask else np.ones_like(xc)
## ---------------------------------------------------------------------------


# density/gravity
rho = 1000.0

flip_sign = False

# Burger and Rossby
Bu, Ro = 1., .1
print(f"Ro={Ro} Bu={Bu}")

r0, r1, r2 = 0.1 * Lx, 0.1 * Lx, 0.14 * Lx

# set coriolis with burger number
f0 =  np.sqrt(g_prime[0, 0, 0] * H[0, 0, 0] / Bu / r0**2)
if flip_sign:
    f0 *= -1
beta = 0
f = f0 + beta * (y - Ly / 2)

# wind forcing, bottom drag
taux = 0.0
tauy = 0.0
bottom_drag_coef = 0.0

## Model Parameters Definition -------------------------------------------

### QG Model
param_qg = {
    "nx": nx,
    "ny": ny,
    "nl": nl,
    "dx": dx,
    "dy": dy,
    "H": H,
    "g_prime": g_prime,
    "f": f,
    "taux": taux,
    "tauy": tauy,
    "bottom_drag_coef": bottom_drag_coef,
    "dtype": dtype,
    "mask": mask,
    "compile": True,
    "slip_coef": 1.0,
    # dt will be modified later after computing CFL condition
    # yet the model must be instantiated for the conversion between
    # pressure and U,V,H to be done and CFL condition to be computed
    "dt": 0.0,
}

### SW Model
param_sw = {
    "nx": nx,
    "ny": ny,
    "nl": nl,
    "dx": dx,
    "dy": dy,
    "H": H,
    "rho": rho,
    "g_prime": g_prime,
    "f": f,
    "taux": taux,
    "tauy": tauy,
    "mask": mask,
    "bottom_drag_coef": bottom_drag_coef,
    "dtype": dtype,
    "slip_coef": 1,
    "compile": True,
    "barotropic_filter": False,
    "barotropic_filter_spectral": False,
    "dt": 0.0,  # time-step (s)
}
## -----------------------------------------------------------------------

## Create QG model -------------------------------------------------------
qg_multilayer = QG(param_qg)
## -----------------------------------------------------------------------


## Initial Perturbation --------------------------------------------------
# create rankine vortex with tripolar perturbation
z = x + 1j * y
theta = np.angle(z)
r =  np.sqrt(x**2 + y**2)
epsilon = 1e-3
r *= 1 + epsilon * np.cos(theta * 3)

def sigmoid(z):
    return 1/(1 + np.exp(-z))
soft_step = lambda x: sigmoid(x / 100)

mask_core = soft_step(r0 - r)

mask_ring = soft_step(r - r1) * soft_step(r2 - r)
vor = 1.0 * (-mask_core / mask_core.mean() + mask_ring / mask_ring.mean())
if flip_sign:
    vor *= -1

laplace_dstI = compute_laplace_dstI(
    nx, ny, dx, dy, {"dtype": dtype}
)
psi_hat = dstI2D(vor[1:-1, 1:-1]) / laplace_dstI
psi = jnp.expand_dims(jnp.expand_dims(
    jnp.pad(dstI2D(psi_hat), ((1, 1), (1, 1))),
    axis=0), axis=0)

# set psi amplitude to have correct Rossby number
u, v = grad_perp(psi, dx, dy)
u_norm_max = max(np.abs(u).max(), np.abs(v).max())
psi *= Ro * f0 * r0 / u_norm_max
p_init = psi * f0
# Use G to convert pressure to u,v and h to initialize the model
# This is why the QG model must be instantiated before the initial
# conditions are defined, to provide the "pressure to U,V,H" function G
u_init, v_init, h_init = qg_multilayer.G(p_init)
## -----------------------------------------------------------------------

u_max, v_max, c = (
    np.abs(u_init).max() / dx,
    np.abs(v_init).max() / dy,
    np.sqrt(g_prime[0, 0, 0] * H.sum()),
)
print(f"u_max {u_max:.2e}, v_max {v_max:.2e}, c {c:.2e}")
cfl_adv = 0.5
cfl_gravity = 5 if param_sw["barotropic_filter"] else 0.5

## Compute dt from U,V and H using the CFL condition ---------------------
dt = min(cfl_adv * dx / u_max, cfl_adv * dy / v_max, cfl_gravity * dx / c)
## -----------------------------------------------------------------------

## Set Time step ---------------------------------------------------------
qg_multilayer.dt = dt
## -----------------------------------------------------------------------

## -----------------------------------------------------------------------
omega = qg_multilayer.compute_omega(u_init, v_init)
w_qg = np.asarray((omega.squeeze() / qg_multilayer.area))

## Set Time step for SW Parameters ---------------------------------------
param_sw["dt"] = dt
## -----------------------------------------------------------------------

## Create SW model -------------------------------------------------------
sw_multilayer = SW(param_sw)
## -----------------------------------------------------------------------

## -----------------------------------------------------------------------

# time params
t = 0
u_a, v_a, k_energy_a, omega_a, div_a = qg_multilayer.compute_ageostrophic_velocity(
    qg_multilayer.compute_time_derivatives(u_init, v_init, h_init),
    sw_multilayer.compute_time_derivatives(u_init, v_init, h_init)
)
wa_0 = np.asarray(omega_a.squeeze())
w_0 = omega.squeeze() / qg_multilayer.dx / qg_multilayer.dy
tau = 1.0 / np.sqrt(jnp.power(w_0, 2).mean())
print(f"tau = {tau * f0:.2f} f0-1")

t_end = 8 * tau
freq_plot = 0#int(t_end / 10 / dt) + 1
freq_checknan = 100
freq_log = int(t_end / 10 / dt) + 1
n_steps = int(t_end / dt) + 1

print(n_steps)
matplotlib.rcParams.update({"font.size": 18})
palette = plt.cm.bwr  # .with_extremes(bad='grey')

u = +u_init / qg_multilayer.dx
v = +v_init / qg_multilayer.dy
h = +h_init / qg_multilayer.area
hl_mean = h.mean((-1,-2)).squeeze()
with np.printoptions(precision=2):
    print(
        f'init, ' \
        f'u: {np.mean(u):+.5E}, ' \
        f'{np.abs(u).max():.5E}, ' \
        f'v: {np.mean(v):+.5E}, ' \
        f'{np.abs(v).max():.5E}, ' \
        f'hl_mean: {hl_mean}, ' \
        f'h min: {h.min():.5E}, ' \
        f'max: {h.max():.5E}, '
    )

for n in range(0, n_steps + 1):
    if False:
        if freq_plot > 0 and (n % freq_plot == 0 or n == n_steps):
            f, a = plt.subplots(1, 3, figsize=(18, 8))
            #a[0].set_title("$\omega_{qg}$")
            #a[1].set_title("$\omega_{sw}$")
            #a[2].set_title("$\omega_{qg} - \omega_{sw}$")
            [(a[i].set_xticks([]), a[i].set_yticks([])) for i in range(3)]
            f.tight_layout()
            mask_w = np.asarray(sw_multilayer.masks.not_w[0, 0])
            w_qg = np.array(qg_multilayer.omega) / qg_multilayer.area / qg_multilayer.f0
            w_sw = sw_multilayer.omega / sw_multilayer.area / sw_multilayer.f0
            wM = max(np.abs(w_qg).max(), np.abs(w_sw).max())

            kwargs = dict(
                cmap=palette, origin="lower", vmin=-wM, vmax=wM, animated=True
            )
            a[0].imshow(np.ma.masked_where(mask_w, w_qg[0, 0]).T, **kwargs)
            a[1].imshow(np.ma.masked_where(mask_w, w_sw[0, 0]).T, **kwargs)
            a[2].imshow(np.ma.masked_where(mask_w, (w_qg - w_sw)[0, 0]).T, **kwargs)
            f.suptitle(
                f"Ro={Ro:.2f}, Bu={Bu:.2f}, t={t / tau:.2f}$\\tau$, "
                f"{'neg.' if flip_sign else 'pos'} $f_0$"
            )
            plt.show()
            plt.pause(0.05)

    ## Model step --------------------------------------------------------
    ### QG
    u,v,h = qg_multilayer.step(u,v,h)
    ### SW
    #u,v,h = qg_multilayer.step(u0,v0,h0)

    ## -------------------------------------------------------------------
    t += dt
    if n % freq_checknan == 0:
        if jnp.isnan(h).any():
            raise ValueError(f"Stopping, NAN number in QG h at iteration {n}.")
        if jnp.isnan(h).any():
            raise ValueError(f"Stopping, NAN number in SW h at iteration {n}.")

    if freq_log > 0 and n % freq_log == 0:
        plt.figure()
        plt.pcolormesh(h[0,0])
        plt.colorbar()
        plt.title(f'jax h at n={n:05d}')
        plt.show()
        plt.pause(0.05)
        hl_mean = h.mean((-1,-2)).squeeze()
        with np.printoptions(precision=2):
            print(
                f'n={n:05d}, ' \
                f'u: {np.mean(u):+.5E}, ' \
                f'{np.abs(u).max():.5E}, ' \
                f'v: {np.mean(v):+.5E}, ' \
                f'{np.abs(v).max():.5E}, ' \
                f'hl_mean: {hl_mean}, ' \
                f'h min: {h.min():.5E}, ' \
                f'max: {h.max():.5E}, '
            )
        #print(f"n={n:05d}, {qg_multilayer.get_print_info()}")