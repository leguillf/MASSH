"""
Shallow-water implementation.
Louis Thiry, Nov 2023 for IFREMER.
"""
import sys 
sys.path.insert(0, '../../src') # add src to path to import modules
from src.config import USE_FLOAT64
import numpy as np
import jax.numpy as jnp 
import jax
from jax import lax
from jax import checkpoint  # same as jax.remat
from jax import jit

from finite_diff import interp_TP, interp_TP_inv, comp_ke, div_nofluxbc
from flux import flux
from helmholtz import HelmholtzNeumannSolver
from masks import Masks
from reconstruction import linear2_centered, wenoz4_left, wenoz6_left
from tools import avg_pool2d

jax.config.update("jax_enable_x64", USE_FLOAT64)


from functools import partial

def smooth_clamp(x, x_min, sharpness=10.):
    """Smooth approximation of jnp.maximum(x, x_min) using softplus.
    Unlike jnp.maximum, the gradient is non-zero everywhere,
    which is critical for adjoint / backward differentiation stability.
    sharpness controls the transition steepness (higher = closer to hard clamp).
    """
    return x_min + jax.nn.softplus((x - x_min) * sharpness) / sharpness


def replicate_pad(f, mask):
    f_ = jnp.pad(f, ((0, 0), (0,0), (1,1), (1,1)), mode='edge')
    mask_ = jnp.pad(mask, ((0, 0), (0,0), (1,1), (1,1)), mode='edge')
    mask_sum = avg_pool2d(
        avg_pool2d(mask_, (3,1), stride=(1,1), padding=(1,0), divisor_override=1),
        (1,3), stride=(1,1), padding=(0,1), divisor_override=1)
    f_sum = avg_pool2d(
        avg_pool2d(f_, (3,1), stride=(1,1), padding=(1,0), divisor_override=1),
        (1,3), stride=(1,1), padding=(0,1), divisor_override=1)
    f_out = f_sum / jnp.maximum(jnp.ones_like(mask_sum), mask_sum)
    return mask_ * f_ + (1 - mask_) * f_out


def reverse_cumsum(x, dim):
    """Pytorch cumsum in the reverse order
    Example:
    reverse_cumsum(torch.arange(1,4), dim=-1)
    >>> tensor([6, 5, 3])
    """

    return x + jnp.sum(x, axis=dim, keepdims=True) - jnp.cumsum(x, axis=dim)


def inv_reverse_cumsum(x, dim):
    """Inverse of reverse cumsum function"""
    neg_diff = -jnp.diff(x, axis=dim)
    x_last = jnp.take(x, indices=[-1], axis=dim)
    return jnp.concatenate([neg_diff, x_last], axis=dim)



class SW:
    """
    # Implementation of multilayer rotating shallow-water model

    Following https://doi.org/10.1029/2021MS002663 .

    ## Main ingredients
        - vector invariant formulation
        - velocity RHS using vortex force upwinding with wenoz-5 reconstruction
        - mass continuity RHS with finite volume using wenoz-5 recontruction

    ## Variables
    Prognostic variables u, v, h differ from physical variables
    u_phys, v_phys (velocity components) and
    h_phys (layer thickness perturbation) as they include
    metric terms dx and dy :
      - u = u_phys x dx
      - v = v_phys x dy
      - h = g_phys x dx x dy

    Diagnostic variables are :
      - U = u_phys / dx
      - V = v_phys / dx
      - omega = omega_phys x dx x dy    (rel. vorticity)
      - eta = eta_phys                  (interface height)
      - p = p_phys                      (hydrostratic pressure)
      - k_energy = k_energy_phys        (kinetic energy)
      - pv = pv_phys                    (potential vorticity)

    ## Time integration
    Explicit time integration with RK3-SSP scheme.

    """

    def __init__(self, param):
        """
        Parameters

        param: python dict. with following keys
            'nx':       int, number of grid points in dimension x
            'ny':       int, number grid points in dimension y
            'nl':       nl, number of stacked layer
            'dx':       float or Tensor (nx, ny), dx metric term
            'dy':       float or Tensor (nx, ny), dy metric term
            'H':        Tensor (nl,) or (nl, nx, ny), unperturbed layer thickness
            'g_prime':  Tensor (nl,), reduced gravities
            'f':        Tensor (nx, ny), Coriolis parameter
            'taux':     float or Tensor (nx-1, ny), top-layer forcing, x component
            'tauy':     float or Tensor (nx, ny-1), top-layer forcing, y component
            'dt':       float > 0., integration time-step
            'n_ens':    int, number of ensemble member
            'dtype':    torch.float32 of torch.float64
            'slip_coef':    float, 1 for free slip, 0 for no-slip, inbetween for
                        partial free slip.
            'bottom_drag_coef': float, linear bottom drag coefficient
            'barotropic_filter': boolean, i true applies implicit FS calculation
        """

        print(f'Creating {self.__class__.__name__} model...')
        self.dtype = param['dtype'] if 'dtype' in param.keys() else jnp.float64
        print(self.dtype)
        self.arr_kwargs = {
            'dtype': self.dtype,
        }

        # verifications
        assert len(param['H'].shape) >= 3, \
            'H must be a nz x ny x nx tensor ' \
            'with nx=1 or ny=1 if H does not vary ' \
            f'in x or y direction, got shape {param["H"].shape}.'

        # grid
        self.nx = param['nx']
        self.ny = param['ny']
        self.nl = param['nl']
        self.dx = param['dx']
        self.dy = param['dy']
        self.H = param['H']
        print(f'  - nx, ny, nl =  {self.nx, self.ny, self.nl}')
        self.area = self.dx*self.dy
        self.slip_coef = param['slip_coef'] if 'slip_coef' in param.keys() else 1.

        # optional mask
        nx, ny = self.nx, self.ny
        if 'mask' in param.keys():
            mask = param['mask']
            shape = mask.shape[0], mask.shape[1]
            assert  shape == (nx, ny), f'Invalid mask shape {shape=}!=({nx},{ny})'
            vals = jnp.unique(mask).tolist()
            assert  all([v in [0,1] for v in vals]) and vals != [0], \
                    f'Invalid mask with non-binary values : {vals}'
            print(f'  - {"non-" if len(vals)==2 else ""}trivial mask provided')

        else:
            print('  - no mask provided, domain assumed to be rectangular')
            mask = jnp.ones((nx, ny), dtype=self.dtype)
        self.masks = Masks(mask)

        # boundary conditions
        assert self.slip_coef >= 0 and self.slip_coef <= 1, \
               f'slip coefficient must be in [0, 1], got {self.slip_coef}'
        cl_type = "free-" if self.slip_coef == 1 else \
                  ("no-" if self.slip_coef == 0 else "partial free-")
        print(f'  - {cl_type}slip boundary condition')

        # Coriolis parameter
        f = param['f']
        shape = f.shape[0], f.shape[1]
        assert  shape == (nx+1, ny+1), f'Invalid f shape {shape=}!=({nx},{ny})'
        self.f = np.expand_dims(f, axis=0)
        self.f0 = self.f.mean()
        self.f_ugrid = 0.5 * (self.f[:,:,1:] + self.f[:,:,:-1])
        self.f_vgrid = 0.5 * (self.f[:,1:,:] + self.f[:,:-1,:])
        self.f_hgrid = interp_TP(self.f)
        self.fstar_ugrid = self.f_ugrid * self.area
        self.fstar_vgrid = self.f_vgrid * self.area
        self.fstar_vgrid = self.f_vgrid * self.area
        self.fstar_hgrid = self.f_hgrid * self.area

        # gravity - reshape for broadcasting
        g_prime = param["g_prime"]
        self.g_prime = g_prime.reshape(-1, 1, 1) if len(g_prime.shape) == 1 else g_prime
        self.g = g_prime[0]

        # external top-layer forcing
        taux, tauy = param['taux'], param['tauy']
        self.set_wind_forcing(taux, tauy)
        self.bottom_drag_coef = param['bottom_drag_coef']
        # Ocean water density (kg/m³) used in wind-stress → acceleration conversion
        # tau [Pa] / (rho_water [kg/m³] × H [m]) × dx [m] gives m²/s² (scaled tendency)
        self.rho_water = param['rho_water'] if 'rho_water' in param else 1025.0

        # Physical layer depth (m) for wind-stress forcing.
        # IMPORTANT for 1-layer QG/SW models: the model's equivalent depth H = c²/g
        # is typically ~0.4–1 m, while the actual mixed-layer depth driving momentum
        # exchange is ~50–200 m.  Setting h_wind to the physical mixed-layer depth
        # gives the correct forcing magnitude.  If None, falls back to the model's
        # reference layer thickness H_ref (the equivalent depth), which is physically
        # correct only for multi-layer models where H represents the true layer depth.
        self.h_wind = param.get('h_wind', None)
        if self.nl == 1 and self.h_wind is None:
            import warnings
            warnings.warn(
                "\n[SW model] nl=1 and h_wind is not set.\n"
                "  The model's equivalent depth H = c²/g ≈ {:.2f} m is used as the\n"
                "  wind-stress denominator, but the physical mixed-layer depth is\n"
                "  typically 50–200 m.  Wind forcing will be ~{:.0f}× too large.\n"
                "  → Set  param['h_wind'] = <mixed_layer_depth_m>  (e.g. 100.)".format(
                    float(self.H.mean()),
                    max(1., 100. / max(float(self.H.mean()), 1e-6))),
                stacklevel=2,
            )

        # Minimum layer thickness to prevent negative h_tot
        self.h_min = param['h_min'] if 'h_min' in param.keys() else 0.1
        self.h_min_sharpness = param['h_min_sharpness'] if 'h_min_sharpness' in param.keys() else 10.

        # Diffusion (Laplacian, in m²/s)
        # visc_coef: velocity diffusion, diff_coef: thickness diffusion
        # Both are critical for adjoint stability with WENO advection.
        self.visc_coef = param['visc_coef'] if 'visc_coef' in param.keys() else 0.
        self.diff_coef = param['diff_coef'] if 'diff_coef' in param.keys() else 0.

        # time
        self.dt = param['dt']
        print(f'  - integration time step {self.dt:.3e}')

        # ensemble
        self.n_ens = param['n_ens'] if 'n_ens' in param.keys() else 1

        # topography and ref values
        self.set_ref_values(self.H)

        # utils and flux computation functions
        self.comp_ke = comp_ke
        self.interp_TP = interp_TP
        self.interp_TP_inv = interp_TP_inv
        self.h_flux_y = lambda h, v: flux(
                h, v,
                dim=-1,
                n_points=6,
                rec_func_2=linear2_centered,
                rec_func_4=wenoz4_left,
                rec_func_6=wenoz6_left,
                mask_2=self.masks.v_sten_hy_eq2[...,1:-1],
                mask_4=self.masks.v_sten_hy_eq4[...,1:-1],
                mask_6=self.masks.v_sten_hy_gt6[...,1:-1])
        self.h_flux_x = lambda h, u: flux(
                h, u,
                dim=-2,
                n_points=6,
                rec_func_2=linear2_centered,
                rec_func_4=wenoz4_left,
                rec_func_6=wenoz6_left,
                mask_2=self.masks.u_sten_hx_eq2[...,1:-1,:],
                mask_4=self.masks.u_sten_hx_eq4[...,1:-1,:],
                mask_6=self.masks.u_sten_hx_gt6[...,1:-1,:])

        self.w_flux_y = lambda w, v_ugrid: flux(
                w, v_ugrid,
                dim=-1,
                n_points=6,
                rec_func_2=linear2_centered,
                rec_func_4=wenoz4_left,
                rec_func_6=wenoz6_left,
                mask_2=self.masks.u_sten_wy_eq2[...,1:-1,:],
                mask_4=self.masks.u_sten_wy_eq4[...,1:-1,:],
                mask_6=self.masks.u_sten_wy_gt4[...,1:-1,:])
        self.w_flux_x = lambda w, u_vgrid: flux(
                w, u_vgrid,
                dim=-2,
                n_points=6,
                rec_func_2=linear2_centered,
                rec_func_4=wenoz4_left,
                rec_func_6=wenoz6_left,
                mask_2=self.masks.v_sten_wx_eq2[...,1:-1],
                mask_4=self.masks.v_sten_wx_eq4[...,1:-1],
                mask_6=self.masks.v_sten_wx_gt6[...,1:-1])

        # barotropic waves filtering for SW
        self.barotropic_filter = False
        if 'barotropic_filter' in param.keys() and param['barotropic_filter']:
            class_name = self.__class__.__name__
            if  class_name == 'SW':
                print('  - Using barotropic filter ', end="")
                self.barotropic_filter = param['barotropic_filter']
                self.tau = 2*self.dt
                if 'barotropic_filter_spectral' in param.keys() and param['barotropic_filter_spectral']:
                    print('spectral approximation')
                    self.barotropic_filter_spectral = True
                    self.H_tot = self.H.sum(axis=-3, keepdims=True)
                    self.lambd = 1. / (self.g * self.dt * self.tau * self.H_tot)
                    self.helm_solver = HelmholtzNeumannSolver(
                            self.nx, self.ny, self.dx, self.dy, self.lambd,
                            self.dtype, mask=self.masks.h[0,0])
                else:
                    self.barotropic_filter_spectral = False
                    print('in exact form')
                    from helmholtz_multigrid import MG_Helmholtz
                    coef_ugrid = (self.h_ref_ugrid * self.masks.u)[0,0]
                    coef_vgrid = (self.h_ref_vgrid * self.masks.v)[0,0]
                    lambd = 1. / (self.g * self.dt * self.tau)
                    self.helm_solver = MG_Helmholtz(self.dx, self.dy,
                            self.nx, self.ny, coef_ugrid, coef_vgrid=coef_vgrid,
                            lambd=lambd, dtype=self.dtype,
                            mask=self.masks.h[0,0], niter_bottom=20,
                            use_compilation=True)
            else:
                print(f'  - class {class_name}!=SW, ignoring barotropic filter ')

        # precompile torch functions
        use_compilation =  param['compile'] if 'compile' in param.keys() else True
        if use_compilation:
            self.comp_ke = jit(self.comp_ke)
            self.interp_TP = jit(self.interp_TP)
            self.h_flux_y = jit(self.h_flux_y)
            self.h_flux_x = jit(self.h_flux_x)
            self.w_flux_y = jit(self.w_flux_y)
            self.w_flux_x = jit(self.w_flux_x)
            self.step = jit(self.step, static_argnames=['nstep'])
            self.step_tgl = jit(self.step_tgl, static_argnames=['nstep'])
            self.step_adj = jit(self.step_adj, static_argnames=['nstep'])
        else:
            print('  - No compilation')

        # Linear / Non Linear
        self.flag_linear = param['flag_linear'] if 'flag_linear' in param.keys() else False

        # Open Boundary conditions
        self.flag_obc = param['flag_obc'] if 'flag_obc' in param.keys() else False
        if self.flag_obc:
            self.obc_kind = param['obc_kind'] if 'obc_kind' in param.keys() else '1d'
            self.uS = jnp.zeros((self.nx+1))
            self.uN = jnp.zeros((self.nx+1))
            self.vW = jnp.zeros((self.ny+1))
            self.vE = jnp.zeros((self.ny+1))
            self.hS = jnp.zeros((self.nx))
            self.hN = jnp.zeros((self.nx))
            self.hW = jnp.zeros((self.ny))
            self.hE = jnp.zeros((self.ny))
            self.masks.v = self.masks.v.at[:,:,:,0].set(1)
            self.masks.v = self.masks.v.at[:,:,:,-1].set(1)
            self.masks.u = self.masks.u.at[:,:,0,:].set(1)
            self.masks.u = self.masks.u.at[:,:,-1,:].set(1)

            print(f'  - Using {self.obc_kind} open boundary condition')
        
        # Sponge BC
        self.sponge_coef = param['sponge_coef'] if 'sponge_coef' in param.keys() else 0.
        self.sponge_u = jnp.zeros((1,1,self.nx+1, self.ny))
        self.sponge_v = jnp.zeros((1,1,self.nx, self.ny+1))
        self.sponge_h = jnp.zeros((1,1,self.nx, self.ny))

        # Momentum forcing mode: 'direct' uses Fu/Fv as given,
        # 'mass_consistent' derives Fu/Fv from Fh so that velocity is
        # conserved when mass is added:  Fu = -u/h * Fh, Fv = -v/h * Fh.
        self.forcing_momentum = param.get('forcing_momentum', 'direct')

        

    def _compute_ref_values(self, H):
        """Pure functional computation of reference values.
        Returns (h_ref, h_ref_ugrid, h_ref_vgrid, dx_p_ref, dy_p_ref).
        No mutation — safe for JAX AD.
        """
        h_ref = H * self.area
        eta_ref = -H.sum(axis=-3) + reverse_cumsum(H, dim=-3)
        p_ref = jnp.cumsum(self.g_prime * eta_ref, axis=-3)
        if h_ref.shape[-2] != 1 and h_ref.shape[-1] != 1:
            _h_ref_u = jnp.pad(h_ref, ((0, 0), (1, 1), (0, 0)), mode='edge')
            h_ref_ugrid = 0.5 * (_h_ref_u[...,1:,:] + _h_ref_u[...,:-1,:])
            _h_ref_v = jnp.pad(h_ref, ((0, 0), (0, 0), (1, 1)), mode='edge')
            h_ref_vgrid = 0.5 * (_h_ref_v[...,1:] + _h_ref_v[...,:-1])
            dx_p_ref = jnp.diff(p_ref, axis=-2)
            dy_p_ref = jnp.diff(p_ref, axis=-1)
        else:
            h_ref_ugrid = h_ref
            h_ref_vgrid = h_ref
            dx_p_ref = 0.
            dy_p_ref = 0.
        return h_ref, h_ref_ugrid, h_ref_vgrid, dx_p_ref, dy_p_ref

    def set_ref_values(self, H):
        self.h_ref, self.h_ref_ugrid, self.h_ref_vgrid, \
            self.dx_p_ref, self.dy_p_ref = self._compute_ref_values(H)
        self.eta_ref = -H.sum(axis=-3) + reverse_cumsum(H, dim=-3)
        self.p_ref = jnp.cumsum(self.g_prime * self.eta_ref, axis=-3)

    def set_wind_forcing(self, taux, tauy):
        nx, ny = self.nx, self.ny
        assert type(taux) == float or taux.shape == (nx-1, ny), \
               f'taux must be a float or a {(nx-1, ny)} Tensor'
        assert type(tauy) == float or tauy.shape == (nx, ny-1), \
               f'tauy must be a float or a {(nx, ny-1)} Tensor'
        self.taux = taux
        self.tauy = tauy

    def get_physical_uvh(self, u, v, h, numpy=False):
        """Get physical variables u_phys, v_phys, h_phys from state variables."""
        u_phys = (u / self.dx)
        v_phys = (v / self.dy)
        h_phys = (h / self.area)

        return (np.array(u_phys), np.array(v_phys), np.array(h_phys)) if numpy \
               else (u_phys, v_phys, h_phys)

    def set_input_uvh(self, u_phys, v_phys, h_phys):
        """
        Set state variables with physical variables u_phys, v_phys, h_phys.
        """
        u_ = jnp.array(u_phys) if isinstance(u_phys, np.ndarray) else u_phys
        v_ = jnp.array(v_phys) if isinstance(v_phys, np.ndarray) else v_phys
        h_ = jnp.array(h_phys) if isinstance(h_phys, np.ndarray) else h_phys

        u_ = u_ * self.masks.u
        v_ = v_ * self.masks.v
        h_ = h_ * self.masks.h 

        u = u_.astype(self.dtype) * self.dx
        v = v_.astype(self.dtype) * self.dy
        h = h_.astype(self.dtype) * self.area

        return u, v, h
        
    def get_print_info(self, u, v, h):
        """
        Returns a string with summary of current variables.
        """
        hl_mean = (h / self.area).mean((-1,-2)).squeeze()
        eta = reverse_cumsum(h / self.area, dim=-3)
        with np.printoptions(precision=2):
            return \
                f'u: {np.mean(u):+.5E}, ' \
                f'{np.abs(u).max():.5E}, ' \
                f'v: {np.mean(v):+.5E}, ' \
                f'{np.abs(v).max():.5E}, ' \
                f'hl_mean: {hl_mean}, ' \
                f'h min: {h.min():.5E}, ' \
                f'max: {h.max():.5E}, ' \
                f'eta_sur min: {eta[:,0].min():+.5f}, ' \
                f'max: {eta[:,0].max():.5f}'

    def advection_h(self, U, V, h, h_ref=None):
        """
        Advection RHS for thickness perturbation h
        dt_h = - div(h_tot [u v]),  h_tot = h_ref + h
        """
        _h_ref = h_ref if h_ref is not None else self.h_ref
        if self.flag_linear:
            h_tot = _h_ref * jnp.ones_like(h)
        else:
            h_tot = _h_ref + h
            h_tot = smooth_clamp(h_tot, self.h_min * self.area, self.h_min_sharpness)
        h_tot_flux_y = self.h_flux_y(h_tot, V[...,1:-1])
        h_tot_flux_x = self.h_flux_x(h_tot, U[...,1:-1,:])
        return -div_nofluxbc(h_tot_flux_x, h_tot_flux_y) * self.masks.h

    def advection_momentum(self, u, v, omega, U_m, V_m, k_energy, p, h_tot_ugrid, h_tot_vgrid,
                           dx_p_ref=None, dy_p_ref=None, taux=None, tauy=None):
        """
        Advection RHS for momentum (u, v)
        """
        _dx_p_ref = dx_p_ref if dx_p_ref is not None else self.dx_p_ref
        _dy_p_ref = dy_p_ref if dy_p_ref is not None else self.dy_p_ref

        # Vortex-force + Coriolis
        omega_Vm = self.w_flux_y(omega[...,1:-1,:], V_m)
        omega_Um = self.w_flux_x(omega[...,1:-1], U_m)

        dt_u = omega_Vm + self.fstar_ugrid[...,1:-1,:] * V_m
        dt_v = -(omega_Um + self.fstar_vgrid[...,1:-1] * U_m)

        # grad pressure + k_energy
        ke_pressure = k_energy + p
        dt_u -= jnp.diff(ke_pressure, axis=-2) + _dx_p_ref
        dt_v -= jnp.diff(ke_pressure, axis=-1) + _dy_p_ref

        # wind forcing and bottom drag
        dt_u, dt_v = self.add_wind_forcing(dt_u, dt_v, h_tot_ugrid, h_tot_vgrid, taux=taux, tauy=tauy)
        dt_u, dt_v = self.add_bottom_drag(dt_u, dt_v, u, v)
        dt_u, dt_v = self.add_diffusion(dt_u, dt_v, u, v)

        return jnp.pad(dt_u, ((0,0), (0,0), (1, 1), (0, 0)))*self.masks.u, \
               jnp.pad(dt_v, ((0,0), (0,0), (0, 0), (1, 1)))*self.masks.v
    
    def add_diffusion(self, du, dv, u, v):
        """
        Add Laplacian diffusion ν∇²(u_phys) to velocity derivatives.
        Uses Neumann (zero-flux) BCs via edge-padding.
        Applies to all layers.
        """
        if self.visc_coef is not None and self.visc_coef > 0:
            # Pad u in y, v in x for Neumann-like boundary treatment
            u_pad = jnp.pad(u, ((0,0), (0,0), (0,0), (1,1)), mode='edge')
            v_pad = jnp.pad(v, ((0,0), (0,0), (1,1), (0,0)), mode='edge')

            # u_phys = u / dx on padded grid
            u_phys = u_pad / self.dx
            v_phys = v_pad / self.dy

            # Laplacian at interior u-points (x: 1:-1, y: all via padding)
            lap_u = (u_phys[..., 2:, 1:-1] - 2*u_phys[..., 1:-1, 1:-1] + u_phys[..., :-2, 1:-1]) / self.dx**2 \
                  + (u_phys[..., 1:-1, 2:] - 2*u_phys[..., 1:-1, 1:-1] + u_phys[..., 1:-1, :-2]) / self.dy**2

            # Laplacian at interior v-points (y: 1:-1, x: all via padding)
            lap_v = (v_phys[..., 2:, 1:-1] - 2*v_phys[..., 1:-1, 1:-1] + v_phys[..., :-2, 1:-1]) / self.dx**2 \
                  + (v_phys[..., 1:-1, 2:] - 2*v_phys[..., 1:-1, 1:-1] + v_phys[..., 1:-1, :-2]) / self.dy**2

            # Convert back to scaled variables
            du = du + self.visc_coef * lap_u * self.dx
            dv = dv + self.visc_coef * lap_v * self.dy

        return du, dv

    def add_h_diffusion(self, h):
        """
        Laplacian diffusion κ∇²(h_phys) for layer thickness.
        Returns diffusion tendency in scaled form (h = h_phys * area).
        Critical for adjoint stability: counteracts anti-diffusivity of
        the adjoint WENO scheme.
        """
        if self.diff_coef is not None and self.diff_coef > 0:
            h_pad = jnp.pad(h, ((0,0), (0,0), (1,1), (1,1)), mode='edge')
            lap_h = (h_pad[..., 2:, 1:-1] - 2*h_pad[..., 1:-1, 1:-1] + h_pad[..., :-2, 1:-1]) / self.dx**2 \
                  + (h_pad[..., 1:-1, 2:] - 2*h_pad[..., 1:-1, 1:-1] + h_pad[..., 1:-1, :-2]) / self.dy**2
            return self.diff_coef * lap_h * self.masks.h
        return jnp.zeros_like(h)

    def add_wind_forcing(self, du, dv, h_tot_ugrid, h_tot_vgrid, taux=None, tauy=None):
        """
        Add wind forcing to the derivatives du, dv.
        taux/tauy: wind stress in Pa (N/m²) on (nx-1, ny) and (nx, ny-1) grids.
        If None, falls back to self.taux / self.tauy.

        Physics:
          du/dt += tau_x / (rho_water * H_ref) * dx   [m²/s²]

        The denominator uses the REFERENCE (time-mean) layer thickness h_ref_ugrid,
        NOT the instantaneous h_tot.  Using h_tot creates a destructive feedback:
        wind thins the layer on the upwind side → 1/h_tot grows → wind stress
        explodes → NaN.  h_ref is time-invariant, so this feedback does not exist.
        (This is standard practice in layered ocean models.)

        Safety:
          - jnp.where ensures land u/v points (mask=0) never evaluate tau/H,
            preventing inf*0=NaN in JAX.
          - H_ref is additionally clamped from below at h_min.
        """
        _taux = taux if taux is not None else self.taux
        _tauy = tauy if tauy is not None else self.tauy

        # Interior ocean masks (trim boundary u/v rows/cols to match interior du/dv)
        mask_u = self.masks.u[..., 1:-1, :]   # (..., nx-1, ny)
        mask_v = self.masks.v[..., :, 1:-1]   # (..., nx,   ny-1)

        # Layer depth used in wind-stress denominator (in metres).
        #
        # Two cases:
        #   h_wind is set  →  use the prescribed physical mixed-layer depth (scalar).
        #                     Required for 1-layer QG/SW models where the model's
        #                     equivalent depth H = c²/g ≈ 0.4–1 m, while the real
        #                     mixed-layer driving momentum exchange is ~50–200 m.
        #   h_wind is None →  use self.h_ref_ugrid (model reference thickness, correct
        #                     for multi-layer models where H is the true layer depth).
        if self.h_wind is not None:
            H_ref_u = float(self.h_wind)
            H_ref_v = float(self.h_wind)
        else:
            # self.h_ref_ugrid shape: (nl, nx+1, ny) for spatially-varying H,
            #                         (nl, 1, 1)     for uniform H.
            # Take top layer (index 0), trim interior only when dim > 1.
            H0_u = self.h_ref_ugrid[0]   # (nx+1, ny) or (1, 1)
            H0_v = self.h_ref_vgrid[0]   # (nx, ny+1) or (1, 1)
            if H0_u.shape[0] > 1:
                H0_u = H0_u[1:-1, :]     # (nx-1, ny)
            if H0_v.shape[1] > 1:
                H0_v = H0_v[:, 1:-1]     # (nx, ny-1)
            H_ref_u = jnp.maximum(H0_u / self.area, self.h_min)
            H_ref_v = jnp.maximum(H0_v / self.area, self.h_min)

        # Wind tendency: jnp.where so land points never compute tau/H (avoids inf*0=NaN)
        wind_u = jnp.where(
            mask_u[..., 0, :, :] > 0.5,
            _taux / (self.rho_water * H_ref_u) * self.dx,
            jnp.zeros_like(du[..., 0, :, :]))
        wind_v = jnp.where(
            mask_v[..., 0, :, :] > 0.5,
            _tauy / (self.rho_water * H_ref_v) * self.dy,
            jnp.zeros_like(dv[..., 0, :, :]))

        du = du.at[..., 0, :, :].set(du[..., 0, :, :] + wind_u)
        dv = dv.at[..., 0, :, :].set(dv[..., 0, :, :] + wind_v)
        return du, dv

    def add_bottom_drag(self, du, dv, u, v):
        """
        Add bottom drag to the derivatives du, dv.
        """
        du = du.at[...,-1,:,:].set(du[...,-1,:,:] - self.bottom_drag_coef * u[...,-1,1:-1,:])
        dv = dv.at[...,-1,:,:].set(dv[...,-1,:,:] - self.bottom_drag_coef * v[...,-1,:,1:-1])
        return du, dv

    def compute_omega(self, u, v):
        """
        Pad u and v using boundary conditions (free-slip, partial free-slip,
        no-slip).
        """
        u_ = jnp.pad(u, ((0, 0), (0, 0), (0, 0), (1, 1)))
        v_ = jnp.pad(v, ((0, 0), (0, 0), (1, 1), (0, 0)))
        dx_v = jnp.diff(v_, axis=-2)
        dy_u = jnp.diff(u_, axis=-1)
        curl_uv = dx_v - dy_u
        alpha = 2 * (1 - self.slip_coef)
        omega = self.masks.w_valid * curl_uv \
              + self.masks.w_cornerout_bound * (1 - self.slip_coef) * curl_uv \
              + self.masks.w_vertical_bound * alpha * dx_v \
              - self.masks.w_horizontal_bound * alpha * dy_u

        return omega

    def compute_diagnostic_variables(self, u, v, h, h_ref_ugrid=None, h_ref_vgrid=None):
        """
        Compute the model's diagnostic variables given the prognostic
        variables self.u, self.v, self.h .
        """
        _h_ref_ugrid = h_ref_ugrid if h_ref_ugrid is not None else self.h_ref_ugrid
        _h_ref_vgrid = h_ref_vgrid if h_ref_vgrid is not None else self.h_ref_vgrid

        omega = self.compute_omega(u, v)
        eta = reverse_cumsum(h / self.area, dim=-3)
        p = jnp.cumsum(self.g_prime * eta, axis=-3)
        U = u / self.dx**2
        V = v / self.dy**2
        U_m = self.interp_TP(U)
        V_m = self.interp_TP(V)
        k_energy = self.comp_ke(u, U, v, V) * self.masks.h
        h_ = replicate_pad(h, self.masks.h)
        h_ugrid = 0.5 * (h_[...,1:,1:-1] + h_[...,:-1,1:-1])
        h_vgrid = 0.5 * (h_[...,1:-1,1:] + h_[...,1:-1,:-1])
        h_tot_ugrid = smooth_clamp(_h_ref_ugrid + h_ugrid, self.h_min * self.area, self.h_min_sharpness)
        h_tot_vgrid = smooth_clamp(_h_ref_vgrid + h_vgrid, self.h_min * self.area, self.h_min_sharpness)

        return omega, eta, p, U, V, U_m, V_m, k_energy, h_tot_ugrid, h_tot_vgrid

    def filter_barotropic_waves(self, dt_u, dt_v, dt_h, u, v, h_tot_ugrid, h_tot_vgrid):
        """
        Inspired from https://doi.org/10.1029/2000JC900089.
        """
        # compute RHS
        u_star = (u + self.dt*dt_u) / self.dx
        v_star = (v + self.dt*dt_v) / self.dy
        u_bar_star = (u_star * h_tot_ugrid).sum(axis=-3, keepdims=True) \
                     / h_tot_ugrid.sum(axis=-3, keepdims=True)
        v_bar_star = (v_star * h_tot_vgrid).sum(axis=-3, keepdims=True) \
                     / h_tot_vgrid.sum(axis=-3, keepdims=True)
        if self.barotropic_filter_spectral:
            rhs = 1. / (self.g * self.dt * self.tau) * (
                    jnp.diff(u_bar_star, axis=-2) / self.dx \
                + jnp.diff(v_bar_star, axis=-1) / self.dy)
            w_surf_imp = self.helm_solver.solve(rhs)
        else:
            rhs = 1. / (self.g * self.dt * self.tau) * (
                    jnp.diff(h_tot_ugrid * u_bar_star, axis=-2) / self.dx \
                  + jnp.diff(h_tot_vgrid * v_bar_star, axis=-1) / self.dy)
            coef_ugrid = (h_tot_ugrid * self.masks.u)[0,0]
            coef_vgrid = (h_tot_vgrid * self.masks.v)[0,0]
            w_surf_imp = self.helm_solver.solve(rhs, coef_ugrid, coef_vgrid)
            # WIP

        filt_u = jnp.pad(-self.g * self.tau * jnp.diff(w_surf_imp, axis=-2), ((0,0), (0,0), (1, 1), (0, 0))) * self.masks.u
        filt_v = jnp.pad(-self.g * self.tau * jnp.diff(w_surf_imp, axis=-1), ((0,0), (0,0), (0, 0), (1, 1))) * self.masks.v

        return dt_u + filt_u, \
               dt_v + filt_v, \
               dt_h

    def compute_time_derivatives(self, u, v, h, ref_vals=None, taux=None, tauy=None):
        """
        Computes the state variables derivatives dt_u, dt_v, dt_h.
        ref_vals: optional tuple (h_ref, h_ref_ugrid, h_ref_vgrid, dx_p_ref, dy_p_ref)
                  for pure-functional usage (needed for correct JAX AD through H).
        taux/tauy: wind stress (overrides self.taux / self.tauy if provided).
        """
        if ref_vals is not None:
            h_ref, h_ref_ugrid, h_ref_vgrid, dx_p_ref, dy_p_ref = ref_vals
        else:
            h_ref = h_ref_ugrid = h_ref_vgrid = dx_p_ref = dy_p_ref = None

        omega, eta, p, U, V, U_m, V_m, k_energy, h_tot_ugrid, h_tot_vgrid = \
            self.compute_diagnostic_variables(u, v, h, h_ref_ugrid, h_ref_vgrid)
        dt_h = self.advection_h(U, V, h, h_ref) + self.add_h_diffusion(h)
        dt_u, dt_v = self.advection_momentum(
            u, v, omega, U_m, V_m, k_energy, p, h_tot_ugrid, h_tot_vgrid,
            dx_p_ref, dy_p_ref, taux=taux, tauy=tauy)
        if self.barotropic_filter:
            dt_u, dt_v, dt_h = self.filter_barotropic_waves(dt_u, dt_v, dt_h, u, v, h_tot_ugrid, h_tot_vgrid)

        return dt_u, dt_v, dt_h

    def step(
        self,
        u0,
        v0,
        h0,
        H=None,
        nstep=1,
        u_b=None,
        v_b=None,
        h_b=None,
        Fu=None,
        Fv=None,
        Fh=None,
        taux=None,
        tauy=None,
    ):
        """
        Performs nstep time-integration with RK3-SSP scheme.
        Memory-efficient for reverse-mode differentiation.
        taux/tauy: wind stress arrays (nx-1, ny) and (nx, ny-1).
        If None, falls back to self.taux / self.tauy set at initialisation.
        Passing them here allows time-varying wind forcing.
        """

        import jax
        import jax.numpy as jnp
        from jax import lax

        # ----------------------------
        # Prepare inputs
        # ----------------------------
        u, v, h = self.set_input_uvh(u0, v0, h0)

        _u_b, _v_b, _h_b = self.set_input_uvh(
            u_b if u_b is not None else u0,
            v_b if v_b is not None else v0,
            h_b if h_b is not None else h0,
        )

        _Fu, _Fv, _Fh = self.set_input_uvh(
            Fu if Fu is not None else jnp.zeros_like(u0),
            Fv if Fv is not None else jnp.zeros_like(v0),
            Fh if Fh is not None else jnp.zeros_like(h0),
        )

        # ---------------------------------------------------
        # Compute ref values FUNCTIONALLY (no self mutation)
        # so JAX AD can differentiate through H correctly.
        # ---------------------------------------------------
        H_total = self.H + H if H is not None else self.H
        ref_vals = self._compute_ref_values(H_total)
        _h_ref = ref_vals[0]  # h_ref for h_floor

        # Resolve wind stress: prefer argument, fall back to self
        _taux = taux if taux is not None else self.taux
        _tauy = tauy if tauy is not None else self.tauy

        # ----------------------------
        # Single RK3 step
        # ----------------------------
        def single_step(carry, _):
            u, v, h = carry

            # Sponge as Rayleigh damping integrated within RK3 substages.
            # Rate γ = sponge_coef / dt  so that ∫₀ᵈᵗ γ ds ≈ sponge_coef.
            _gamma_u = (self.sponge_coef / self.dt) * self.sponge_u
            _gamma_v = (self.sponge_coef / self.dt) * self.sponge_v
            _gamma_h = (self.sponge_coef / self.dt) * self.sponge_h

            # ---- RK3-SSP with sponge damping ----
            dt0_u, dt0_v, dt0_h = self.compute_time_derivatives(u, v, h, ref_vals, taux=_taux, tauy=_tauy)
            dt0_u = dt0_u + _gamma_u * (_u_b - u)
            dt0_v = dt0_v + _gamma_v * (_v_b - v)
            dt0_h = dt0_h + _gamma_h * (_h_b - h)
            u = u + self.dt * dt0_u
            v = v + self.dt * dt0_v
            h = h + self.dt * dt0_h

            dt1_u, dt1_v, dt1_h = self.compute_time_derivatives(u, v, h, ref_vals, taux=_taux, tauy=_tauy)
            dt1_u = dt1_u + _gamma_u * (_u_b - u)
            dt1_v = dt1_v + _gamma_v * (_v_b - v)
            dt1_h = dt1_h + _gamma_h * (_h_b - h)
            u = u + (self.dt / 4.0) * (dt1_u - 3.0 * dt0_u)
            v = v + (self.dt / 4.0) * (dt1_v - 3.0 * dt0_v)
            h = h + (self.dt / 4.0) * (dt1_h - 3.0 * dt0_h)

            dt2_u, dt2_v, dt2_h = self.compute_time_derivatives(u, v, h, ref_vals, taux=_taux, tauy=_tauy)
            dt2_u = dt2_u + _gamma_u * (_u_b - u)
            dt2_v = dt2_v + _gamma_v * (_v_b - v)
            dt2_h = dt2_h + _gamma_h * (_h_b - h)
            u = u + (self.dt / 12.0) * (8.0 * dt2_u - dt1_u - dt0_u)
            v = v + (self.dt / 12.0) * (8.0 * dt2_v - dt1_v - dt0_v)
            h = h + (self.dt / 12.0) * (8.0 * dt2_h - dt1_h - dt0_h)

            # ---- External forcing ----
            if self.forcing_momentum == 'mass_consistent':
                # Derive momentum forcing from mass forcing so that
                # velocity is conserved:  Fu = -u/h_tot * Fh
                h_ = replicate_pad(h, self.masks.h)
                h_ugrid = 0.5 * (h_[..., 1:, 1:-1] + h_[..., :-1, 1:-1])
                h_vgrid = 0.5 * (h_[..., 1:-1, 1:] + h_[..., 1:-1, :-1])
                h_tot_u = smooth_clamp(ref_vals[1] + h_ugrid,
                                       self.h_min * self.area,
                                       self.h_min_sharpness)
                h_tot_v = smooth_clamp(ref_vals[2] + h_vgrid,
                                       self.h_min * self.area,
                                       self.h_min_sharpness)
                Fh_ = replicate_pad(_Fh, self.masks.h)
                Fh_u = 0.5 * (Fh_[..., 1:, 1:-1] + Fh_[..., :-1, 1:-1])
                Fh_v = 0.5 * (Fh_[..., 1:-1, 1:] + Fh_[..., 1:-1, :-1])
                u = u + self.dt * (-u / h_tot_u * Fh_u)
                v = v + self.dt * (-v / h_tot_v * Fh_v)
            else:
                u = u + self.dt * _Fu
                v = v + self.dt * _Fv
            h = h + self.dt * _Fh

            return (u, v, h), None

        # --------------------------------------
        # 🔥 CRITICAL: checkpoint the step
        # --------------------------------------
        single_step = jax.checkpoint(single_step)

        # --------------------------------------
        # Use scan instead of fori_loop
        # --------------------------------------
        if nstep > 0:
            (u, v, h), _ = lax.scan(
                single_step,
                (u, v, h),
                None,
                length=nstep,
            )

        # ----------------------------
        # Back to physical space
        # ----------------------------
        u_phys, v_phys, h_phys = self.get_physical_uvh(
            u, v, h, numpy=False
        )

        return u_phys, v_phys, h_phys

    def step_tgl(self, u0, v0, h0, du0, dv0, dh0, H=None, dH=None, nstep=1, taux=None, tauy=None):
        """
        Tangent Linear Model: computes the linearized evolution of perturbations.
        taux/tauy: wind stress passed through to step().
        """
        def wrapped_step(x):
            u0, v0, h0, H = x
            return self.step(u0, v0, h0, H, nstep=nstep, taux=taux, tauy=tauy)

        primals = ((u0, v0, h0, H),)
        tangents = ((du0, dv0, dh0, dH),)

        y, dy = jax.jvp(wrapped_step, primals, tangents)

        return dy  # returns (du, dv, dh)

    def step_adj(self, u0, v0, h0, wuT, wvT, whT, H=None, nstep=1, taux=None, tauy=None):
        """
        Adjoint Model: computes the adjoint propagation backward.
        taux/tauy: wind stress passed through to step().
        """
        def wrapped_step(x):
            u0, v0, h0, H = x
            return self.step(u0, v0, h0, H, nstep=nstep, taux=taux, tauy=tauy)
        primals = ((u0, v0, h0, H),)
        cotangents = (wuT, wvT, whT)  

        y, vjp_fn = jax.vjp(wrapped_step, *primals)
        adjoints = vjp_fn(cotangents)
        return adjoints  # returns (adj_u0, adj_v0, adj_h0, adj_H)


    def adjoint_test_sw(self, nstep=1, seed=42):
        """
        Low-level adjoint test for the SW model.

        Checks the identity:  <M dx, y> == <dx, M* y>
        where M = step_tgl (tangent-linear) and M* = step_adj (adjoint).

        All random vectors are generated directly in the model's working dtype
        so there is no precision mismatch between what the operators see and
        what the inner products use.

        Uses a state at rest (zeros) as the base trajectory to avoid nonlinear
        blowup with random fields.  Perturbation and cotangent vectors are
        small-amplitude, masked to ocean points.

        Parameters
        ----------
        model : SW instance (already initialised)
        nstep : number of time steps
        seed  : random seed for reproducibility
        """
        key = jax.random.PRNGKey(seed)
        dtype = self.dtype
        n_ens = self.n_ens
        nl = self.nl
        nx = self.nx
        ny = self.ny

        def rand(key, shape):
            key, subkey = jax.random.split(key)
            return key, jax.random.normal(subkey, shape=shape, dtype=dtype) * 1e-4

        # Shapes (physical space): u(n_ens, nl, nx+1, ny), v(n_ens, nl, nx, ny+1), h(n_ens, nl, nx, ny)
        u_shape = (n_ens, nl, nx + 1, ny)
        v_shape = (n_ens, nl, nx, ny + 1)
        h_shape = (n_ens, nl, nx, ny)

        # Base trajectory: state at rest (zero physical perturbation)
        #u0 = jnp.zeros(u_shape, dtype=dtype)
        #v0 = jnp.zeros(v_shape, dtype=dtype)
        #h0 = jnp.zeros(h_shape, dtype=dtype)
        key, u0 = rand(key, u_shape)
        key, v0 = rand(key, v_shape)
        key, h0 = rand(key, h_shape)
        u0 = u0 * self.masks.u
        v0 = v0 * self.masks.v
        h0 = h0 * self.masks.h

        # TLM perturbation (small, masked to ocean)
        key, du0 = rand(key, u_shape)
        key, dv0 = rand(key, v_shape)
        key, dh0 = rand(key, h_shape)
        du0 = du0 * self.masks.u
        dv0 = dv0 * self.masks.v
        dh0 = dh0 * self.masks.h

        # ADJ cotangent (small, masked to ocean)
        key, wu = rand(key, u_shape)
        key, wv = rand(key, v_shape)
        key, wh = rand(key, h_shape)
        wu = wu * self.masks.u
        wv = wv * self.masks.v
        wh = wh * self.masks.h

        # Run TLM:  (du1, dv1, dh1) = M (du0, dv0, dh0)
        du1, dv1, dh1 = self.step_tgl(u0, v0, h0, du0, dv0, dh0, nstep=nstep)

        # Run ADJ:  ((au0, av0, ah0, aH),) = M* (wu, wv, wh)
        adjoints = self.step_adj(u0, v0, h0, wu, wv, wh, nstep=nstep)
        au0, av0, ah0, _aH = adjoints[0]

        # Check for NaN
        has_nan = (jnp.any(jnp.isnan(du1)) or jnp.any(jnp.isnan(dv1)) or
                jnp.any(jnp.isnan(dh1)) or jnp.any(jnp.isnan(au0)) or
                jnp.any(jnp.isnan(av0)) or jnp.any(jnp.isnan(ah0)))
        if has_nan:
            print(f'  SW adjoint test (dtype={dtype}, {nstep=}): NaN detected!')
            print(f'    TLM NaN: du1={jnp.any(jnp.isnan(du1))}, '
                f'dv1={jnp.any(jnp.isnan(dv1))}, dh1={jnp.any(jnp.isnan(dh1))}')
            print(f'    ADJ NaN: au0={jnp.any(jnp.isnan(au0))}, '
                f'av0={jnp.any(jnp.isnan(av0))}, ah0={jnp.any(jnp.isnan(ah0))}')
            return float('nan')

        # Inner products (computed in f64 for accurate accumulation)
        ps1 = (jnp.sum(du1.astype(jnp.float64) * wu.astype(jnp.float64))
            + jnp.sum(dv1.astype(jnp.float64) * wv.astype(jnp.float64))
            + jnp.sum(dh1.astype(jnp.float64) * wh.astype(jnp.float64)))

        ps2 = (jnp.sum(du0.astype(jnp.float64) * au0.astype(jnp.float64))
            + jnp.sum(dv0.astype(jnp.float64) * av0.astype(jnp.float64))
            + jnp.sum(dh0.astype(jnp.float64) * ah0.astype(jnp.float64)))

        ratio = float(ps1 / ps2)
        print(f'  SW adjoint test (dtype={dtype}, {nstep=}): '
            f'<Mdx,y>/<dx,M*y> = {ratio}')
        return ratio
