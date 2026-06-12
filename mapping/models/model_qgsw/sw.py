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
from reconstruction import linear2_centered, linear4_left, linear6_left, smooth_abs, wenoz4_left, wenoz6_left
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
        # NOTE: SW uses (nl, nx, ny) shape (last axis = y); see class docstring
        # and flux operators which use dim=-1 for y and dim=-2 for x.
        assert len(param['H'].shape) >= 3, \
            'H must be a (nl, nx, ny) tensor ' \
            '(use nx=1 or ny=1 to broadcast a single x- or y-column), ' \
            f'got shape {param["H"].shape}.'

        # grid
        self.nx = param['nx']
        self.ny = param['ny']
        self.nl = param['nl']
        self.dx = jnp.asarray(param['dx'], dtype=self.dtype)
        self.dy = jnp.asarray(param['dy'], dtype=self.dtype)
        self.H = param['H']
        print(f'  - nx, ny, nl =  {self.nx, self.ny, self.nl}')
        self.area = self.dx*self.dy

        # Metrics interpolated to u-grid (nx+1, ny) and v-grid (nx, ny+1)
        if self.dx.ndim >= 2:
            _dx_xpad = jnp.pad(self.dx, ((1, 1), (0, 0)), mode='edge')
            _dy_xpad = jnp.pad(self.dy, ((1, 1), (0, 0)), mode='edge')
            self.dx_ugrid = 0.5 * (_dx_xpad[1:, :] + _dx_xpad[:-1, :])
            self.dy_ugrid = 0.5 * (_dy_xpad[1:, :] + _dy_xpad[:-1, :])
            _dx_ypad = jnp.pad(self.dx, ((0, 0), (1, 1)), mode='edge')
            _dy_ypad = jnp.pad(self.dy, ((0, 0), (1, 1)), mode='edge')
            self.dx_vgrid = 0.5 * (_dx_ypad[:, 1:] + _dx_ypad[:, :-1])
            self.dy_vgrid = 0.5 * (_dy_ypad[:, 1:] + _dy_ypad[:, :-1])
        else:
            self.dx_ugrid = self.dx
            self.dy_ugrid = self.dy
            self.dx_vgrid = self.dx
            self.dy_vgrid = self.dy
        self.area_ugrid = self.dx_ugrid * self.dy_ugrid
        self.area_vgrid = self.dx_vgrid * self.dy_vgrid

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
        self.fstar_ugrid = self.f_ugrid * self.area_ugrid
        self.fstar_vgrid = self.f_vgrid * self.area_vgrid
        self.fstar_vgrid = self.f_vgrid * self.area_vgrid
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
        self.h_min = param['h_min'] if 'h_min' in param.keys() else 0.01
        self.h_min_sharpness = param['h_min_sharpness'] if 'h_min_sharpness' in param.keys() else 5.

        # Equivalent depth bounds
        self.H_min = param['H_min'] if 'H_min' in param.keys() else None
        self.H_max = param['H_max'] if 'H_max' in param.keys() else None

        # Diffusion (Laplacian, in m²/s)
        # visc_coef: velocity diffusion, diff_coef: thickness diffusion
        # Both are critical for adjoint stability with WENO advection.
        self.visc_coef = param['visc_coef'] if 'visc_coef' in param.keys() else 0.
        self.diff_coef = param['diff_coef'] if 'diff_coef' in param.keys() else 0.
        self.time_scheme    = param.get('time_scheme',    'rk3')   # 'rk3' | 'rk2' | 'rk2_ssp'
        self.h_adv_scheme   = param.get('h_adv_scheme',   'weno')  # 'weno' | 'linear_upwind3' | 'linear_upwind5' | 'rusanov1'/'upwind1'
        self.mom_adv_scheme = param.get('mom_adv_scheme', 'weno')  # 'weno' | 'upwind3' | 'upwind5'

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
            self.step_with_tracer = jit(self.step_with_tracer, static_argnames=['nstep'])
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

        # Tracer diffusivity (m^2/s) — mirrors diff_coef for h
        self.tracer_diff_coef = param.get('diff_coef_trac', 0.)
        self.tracer_adv_scheme = param.get('tracer_adv_scheme', 'weno')  # 'weno' | 'linear_upwind3' | 'linear_upwind5' | 'rusanov1'/'upwind1'

        # Momentum forcing mode: 'direct' uses Fu/Fv as given,
        # 'mass_consistent' derives Fu/Fv from Fh so that velocity is
        # conserved when mass is added:  Fu = -u/h * Fh, Fv = -v/h * Fh.
        self.forcing_momentum = param.get('forcing_momentum', 'direct')

        # Adjoint checkpoint strategy for lax.scan body:
        #   'full'       – checkpoint entire single_step (current default, low memory)
        #   'stage'      – checkpoint only compute_time_derivatives; save RK3 stage
        #                  states as scan residuals (~6× more memory, faster adjoint)
        #   'custom_vjp' – explicit RK3 adjoint via custom_vjp; saves only the 3
        #                  stage states as residuals (minimum memory + fast adjoint)
        # checkpoint_mode is fixed to 'full' (stage/custom_vjp benched and removed)

        # Set to False to disable checkpoint on the scan body (stores full trajectory
        # in device memory). Benchmarked 2026-05-27: NOT viable for qgsw — WENO3+RK3
        # intermediates cause OOM at nstep=72 and slower adjoint at nstep=10 due to
        # HBM bandwidth pressure. Keep True (current default is optimal).
        self.scan_checkpoint = True

    def _compute_ref_values(self, H):
        """Pure functional computation of reference values.
        Returns (h_ref, h_ref_ugrid, h_ref_vgrid, dx_p_ref, dy_p_ref).
        No mutation — safe for JAX AD.
        """
        h_ref = H * self.area
        eta_ref = -H.sum(axis=-3) + reverse_cumsum(H, dim=-3)
        p_ref = jnp.cumsum(self.g_prime * eta_ref, axis=-3)

        # When H is spatially uniform (nx=1, ny=1), padding the 1-element
        # spatial axes produces wrong shapes: (nl,1,1) → (nl,3,1) → h_ref_ugrid
        # (nl,2,1), which then fails to broadcast with h_ugrid (1,nl,nx+1,ny).
        # For uniform H, skip the stagger and use H * area on each staggered
        # grid directly. When dx/dy are scalars (uniform grid) this is identical
        # to the old h_ref branch and broadcasts freely. When dx/dy are 2D
        # (geographic grid) area has shape (nx,ny) so h_ref=(1,nx,ny) which
        # does NOT broadcast with h_ugrid=(1,1,nx+1,ny) due to the x-size
        # mismatch. Using area_ugrid/(nx+1,ny) and area_vgrid/(nx,ny+1) gives
        # the correct shapes in both cases.
        if H.shape[-2] == 1 and H.shape[-1] == 1:
            h_ref_ugrid = H * self.area_ugrid
            h_ref_vgrid = H * self.area_vgrid
        else:
            _h_ref_u = jnp.pad(h_ref, ((0, 0), (1, 1), (0, 0)), mode='edge')
            h_ref_ugrid = 0.5 * (_h_ref_u[...,1:,:] + _h_ref_u[...,:-1,:])
            _h_ref_v = jnp.pad(h_ref, ((0, 0), (0, 0), (1, 1)), mode='edge')
            h_ref_vgrid = 0.5 * (_h_ref_v[...,1:] + _h_ref_v[...,:-1])

        # Compute reference pressure gradients per dimension independently.
        # The previous AND condition (shape[-2]!=1 AND shape[-1]!=1) missed
        # the case where H varies in only one spatial dimension (nl>1).
        dx_p_ref = jnp.diff(p_ref, axis=-2) if H.shape[-2] != 1 else 0.
        dy_p_ref = jnp.diff(p_ref, axis=-1) if H.shape[-1] != 1 else 0.

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
        u_phys = (u / self.dx_ugrid)
        v_phys = (v / self.dy_vgrid)
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

        u_ = jnp.where(self.masks.u > 0.5, u_, 0.0)
        v_ = jnp.where(self.masks.v > 0.5, v_, 0.0)
        h_ = jnp.where(self.masks.h > 0.5, h_, 0.0)

        u = u_.astype(self.dtype) * self.dx_ugrid
        v = v_.astype(self.dtype) * self.dy_vgrid
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

    def advection_tracer(self, U, V, c_area):
        """
        Advective-form RHS for a passive scalar tracer on the h-grid.

        c_area: (1, n_trac, nx, ny)  tracer scaled by cell area
                (c_area = c_phys * area — same storage convention as h)
        U: (1, nl, nx+1, ny)  u_phys/dx  from compute_diagnostic_variables
        V: (1, nl, nx, ny+1)  v_phys/dy  from compute_diagnostic_variables

        Uses the surface velocity (layer 0) for all tracer layers, and reuses
        the same WENO h-grid flux machinery as advection_h.

        Equation form (why this differs from advection_h):
        For mass (h), the conserved quantity is `h_phys * area` and the
        correct SW continuity is `dt(h*area) = -div(area * u_phys * h_face)`.
        For a *passive concentration* (°C, PSU), the conserved quantity is
        `h * c` (not `c` itself). Using the same flux-divergence form as h,
        i.e. `dt(c*area) = -div(area * u_phys * c_face)`, expands to
            dt(c_phys) = -(u·∇c)  -  c_phys * div(u_phys)
        The second term is identically zero for QG geostrophic flow
        (div ≈ 0) but NOT for SW: it acts as a spurious source/sink at
        velocity gradients (jets, fronts), driving c above/below its
        initial range — e.g. negative SSS in convergence zones.

        Fix (advective form): keep the WENO upwind reconstruction of
        `c_phys` at faces (it provides monotone, non-oscillatory upwind
        differencing) but subtract back the spurious `c · div(u_phys)`
        term. Algebraically:
            dt(c_area)_advective = dt(c_area)_flux_div  +  c_phys * div(area·u_phys)
        which is the discrete analogue of `dt(c_phys) = -u·∇c`. This
        preserves spatial constants exactly and prevents excursions
        outside the initial [min, max] range from divergence effects.

        Implementation note (float32 stability):
        Tracer absolute values are O(30) (°C, PSU). Applying WENO to the
        area-scaled `c_area ~ 3e9` overflows float32 inside the WENO-Z
        smoothness weights (β² ~ 1e38). We therefore reconstruct on
        `c_phys = c_area / area` and re-scale by face area outside the
        reconstruction (math-identical for uniform grid spacing,
        FV-consistent for non-uniform area). Output stays area-scaled.
        """
        # Surface velocity: select layer 0, broadcast over tracer axis via (1,1,...)
        U_surf = U[:, 0:1, :, :]   # (1, 1, nx+1, ny)
        V_surf = V[:, 0:1, :, :]   # (1, 1, nx, ny+1)
        # Physical tracer — small magnitudes safe for WENO in float32
        c_phys = c_area / self.area
        # Face-averaged cell area on u- and v-faces (guard for scalar/0-d area).
        if self.area.ndim >= 2:
            area_x = 0.5 * (self.area[1:, :] + self.area[:-1, :])   # (nx-1, ny)
            area_y = 0.5 * (self.area[:, 1:] + self.area[:, :-1])   # (nx, ny-1)
        else:
            area_x = self.area
            area_y = self.area
        # Area-scaled velocity divergence at h-cells: area * div(u_phys).
        # Needed for advective-form correction regardless of scheme.
        vel_div_area = div_nofluxbc(area_x * U_surf[..., 1:-1, :],
                                    area_y * V_surf[..., 1:-1])   # (1, 1, nx, ny)
        if self.tracer_adv_scheme in ('rusanov1', 'upwind1'):
            c_flux_y = area_y * self._h_flux_rusanov1(c_phys, V_surf[..., 1:-1], dim=-1)
            c_flux_x = area_x * self._h_flux_rusanov1(c_phys, U_surf[..., 1:-1, :], dim=-2)
            dt_c_fluxdiv = -div_nofluxbc(c_flux_x, c_flux_y)
            return (dt_c_fluxdiv + c_phys * vel_div_area) * self.masks.h
        if self.tracer_adv_scheme in ('linear_upwind3', 'linear3'):
            c_flux_y = area_y * flux(
                c_phys, V_surf[..., 1:-1], dim=-1, n_points=4,
                rec_func_2=linear2_centered, rec_func_4=linear4_left,
                rec_func_6=linear6_left,
                mask_2=self.masks.v_sten_hy_eq2[..., 1:-1],
                mask_4=self.masks.v_sten_hy_eq4[..., 1:-1],
                mask_6=self.masks.v_sten_hy_gt6[..., 1:-1])
            c_flux_x = area_x * flux(
                c_phys, U_surf[..., 1:-1, :], dim=-2, n_points=4,
                rec_func_2=linear2_centered, rec_func_4=linear4_left,
                rec_func_6=linear6_left,
                mask_2=self.masks.u_sten_hx_eq2[..., 1:-1, :],
                mask_4=self.masks.u_sten_hx_eq4[..., 1:-1, :],
                mask_6=self.masks.u_sten_hx_gt6[..., 1:-1, :])
            dt_c_fluxdiv = -div_nofluxbc(c_flux_x, c_flux_y)
            return (dt_c_fluxdiv + c_phys * vel_div_area) * self.masks.h
        if self.tracer_adv_scheme in ('linear_upwind5', 'linear5'):
            c_flux_y = area_y * flux(
                c_phys, V_surf[..., 1:-1], dim=-1, n_points=6,
                rec_func_2=linear2_centered, rec_func_4=linear4_left,
                rec_func_6=linear6_left,
                mask_2=self.masks.v_sten_hy_eq2[..., 1:-1],
                mask_4=self.masks.v_sten_hy_eq4[..., 1:-1],
                mask_6=self.masks.v_sten_hy_gt6[..., 1:-1])
            c_flux_x = area_x * flux(
                c_phys, U_surf[..., 1:-1, :], dim=-2, n_points=6,
                rec_func_2=linear2_centered, rec_func_4=linear4_left,
                rec_func_6=linear6_left,
                mask_2=self.masks.u_sten_hx_eq2[..., 1:-1, :],
                mask_4=self.masks.u_sten_hx_eq4[..., 1:-1, :],
                mask_6=self.masks.u_sten_hx_gt6[..., 1:-1, :])
            dt_c_fluxdiv = -div_nofluxbc(c_flux_x, c_flux_y)
            return (dt_c_fluxdiv + c_phys * vel_div_area) * self.masks.h
        # WENO upwind reconstruction of c_phys at faces times face velocity.
        c_flux_y = area_y * self.h_flux_y(c_phys, V_surf[..., 1:-1])   # (1, n_trac, nx, ny-1)
        c_flux_x = area_x * self.h_flux_x(c_phys, U_surf[..., 1:-1, :]) # (1, n_trac, nx-1, ny)
        # Flux-divergence tendency (would be correct for h*c, contains
        # spurious c*div(u) term for c alone).
        dt_c_fluxdiv = -div_nofluxbc(c_flux_x, c_flux_y)
        # Cancel spurious c*div(u_phys) source -> pure advective form.
        return (dt_c_fluxdiv + c_phys * vel_div_area) * self.masks.h

    def add_tracer_diffusion(self, c_area):
        """
        Laplacian diffusion \u03ba\u2207\u00b2(c_phys) for passive tracers.
        Mirrors add_h_diffusion: operates on c_phys = c_area/area so that the
        Laplacian is correct on non-uniform grids, then converts back.
        Returns diffusion tendency in area-scaled form.
        """
        if self.tracer_diff_coef is not None and self.tracer_diff_coef > 0:
            c_phys = c_area / self.area
            c_phys_pad = jnp.pad(c_phys, ((0,0),(0,0),(1,1),(1,1)), mode='edge')
            lap_c_phys = (
                (c_phys_pad[..., 2:, 1:-1] - 2*c_phys_pad[..., 1:-1, 1:-1] + c_phys_pad[..., :-2, 1:-1]) / self.dx**2
                + (c_phys_pad[..., 1:-1, 2:] - 2*c_phys_pad[..., 1:-1, 1:-1] + c_phys_pad[..., 1:-1, :-2]) / self.dy**2
            )
            return self.tracer_diff_coef * lap_c_phys * self.area * self.masks.h
        return jnp.zeros_like(c_area)

    def advection_h(self, U, V, h, h_ref=None):
        """
        Advection RHS for thickness perturbation h
        dt_h = - div(h_tot [u v]),  h_tot = h_ref + h

        Implementation note (float32 stability):
        h is stored area-scaled (h_phys * area). With area ~1e8 m² and
        h_phys ~ O(1 m), h_area ~ 1e8. WENO smoothness indicators β ~
        (Δh_area)² ~ 1e14-1e15, and the WENO-Z `smooth_abs(β1-β3)` squares
        them again → ~1e30. Float32 does not overflow here (max 3.4e38) but
        the dynamic range is severely compressed, hurting adjoint
        conditioning (1/(β+ε)^k explodes when β is noisy from catastrophic
        cancellation of two large values).

        Fix (mirrors advection_tracer): apply WENO to h_tot_phys =
        h_tot_area / area (O(1)), then re-scale the reconstructed flux by
        face-averaged area. Math-identical for uniform grid spacing and
        FV-consistent on non-uniform area. Output is still area-scaled
        (d(h_area)/dt = area * d(h_phys)/dt).
        """
        _h_ref = h_ref if h_ref is not None else self.h_ref
        if self.flag_linear:
            h_tot = _h_ref * jnp.ones_like(h)
        else:
            h_tot = _h_ref + h
        # Physical thickness — small magnitudes safe for WENO in float32.
        # Clamp in PHYSICAL units so that h_min_sharpness has a grid-independent
        # meaning: transition width = 1/h_min_sharpness in metres (default 10
        # → 0.1 m).  Previously the clamp acted on the area-scaled h_tot, so
        # for area ~1e8 the kink was effectively a hard step, producing a
        # delta-like adjoint singularity at h_phys ≈ h_min.
        h_tot_phys = h_tot / self.area
        if not self.flag_linear:
            h_tot_phys = smooth_clamp(h_tot_phys, self.h_min, self.h_min_sharpness)
        # Face-averaged cell area on u- and v-faces.
        # Guard: self.area is 0-d (scalar) when dx/dy are uniform scalars (QG);
        # slicing a 0-d array raises IndexError.  For uniform grids face area == cell area.
        if self.area.ndim >= 2:
            area_x = 0.5 * (self.area[1:, :] + self.area[:-1, :])   # (nx-1, ny)
            area_y = 0.5 * (self.area[:, 1:] + self.area[:, :-1])   # (nx, ny-1)
        else:
            area_x = self.area
            area_y = self.area
        if self.h_adv_scheme in ('rusanov1', 'upwind1'):
            h_tot_flux_y = area_y * self._h_flux_rusanov1(h_tot_phys, V[..., 1:-1], dim=-1)
            h_tot_flux_x = area_x * self._h_flux_rusanov1(h_tot_phys, U[..., 1:-1, :], dim=-2)
            return -div_nofluxbc(h_tot_flux_x, h_tot_flux_y) * self.masks.h
        if self.h_adv_scheme in ('linear_upwind3', 'linear3'):
            h_tot_flux_y = area_y * flux(
                h_tot_phys, V[..., 1:-1], dim=-1, n_points=4,
                rec_func_2=linear2_centered, rec_func_4=linear4_left,
                rec_func_6=linear6_left,
                mask_2=self.masks.v_sten_hy_eq2[..., 1:-1],
                mask_4=self.masks.v_sten_hy_eq4[..., 1:-1],
                mask_6=self.masks.v_sten_hy_gt6[..., 1:-1])
            h_tot_flux_x = area_x * flux(
                h_tot_phys, U[..., 1:-1, :], dim=-2, n_points=4,
                rec_func_2=linear2_centered, rec_func_4=linear4_left,
                rec_func_6=linear6_left,
                mask_2=self.masks.u_sten_hx_eq2[..., 1:-1, :],
                mask_4=self.masks.u_sten_hx_eq4[..., 1:-1, :],
                mask_6=self.masks.u_sten_hx_gt6[..., 1:-1, :])
            return -div_nofluxbc(h_tot_flux_x, h_tot_flux_y) * self.masks.h
        if self.h_adv_scheme in ('linear_upwind5', 'linear5'):
            h_tot_flux_y = area_y * flux(
                h_tot_phys, V[..., 1:-1], dim=-1, n_points=6,
                rec_func_2=linear2_centered, rec_func_4=linear4_left,
                rec_func_6=linear6_left,
                mask_2=self.masks.v_sten_hy_eq2[..., 1:-1],
                mask_4=self.masks.v_sten_hy_eq4[..., 1:-1],
                mask_6=self.masks.v_sten_hy_gt6[..., 1:-1])
            h_tot_flux_x = area_x * flux(
                h_tot_phys, U[..., 1:-1, :], dim=-2, n_points=6,
                rec_func_2=linear2_centered, rec_func_4=linear4_left,
                rec_func_6=linear6_left,
                mask_2=self.masks.u_sten_hx_eq2[..., 1:-1, :],
                mask_4=self.masks.u_sten_hx_eq4[..., 1:-1, :],
                mask_6=self.masks.u_sten_hx_gt6[..., 1:-1, :])
            return -div_nofluxbc(h_tot_flux_x, h_tot_flux_y) * self.masks.h
        # WENO fluxes on h_tot_phys, then re-scale by face area
        h_tot_flux_y = area_y * self.h_flux_y(h_tot_phys, V[..., 1:-1])
        h_tot_flux_x = area_x * self.h_flux_x(h_tot_phys, U[..., 1:-1, :])
        return -div_nofluxbc(h_tot_flux_x, h_tot_flux_y) * self.masks.h

    def _h_flux_rusanov1(self, h_tot_phys, velocity, dim):
        """
        First-order conservative upwind/Rusanov flux for h-continuity.

        This is more diffusive than WENO, but the flux is monotone and avoids
        WENO's nonlinear smoothness weights, which can make adjoints fragile.
        `smooth_abs` keeps the local wave speed differentiable at zero velocity.
        """
        h_left, h_right = (
            (h_tot_phys[..., :, :-1], h_tot_phys[..., :, 1:])
            if dim == -1 else
            (h_tot_phys[..., :-1, :], h_tot_phys[..., 1:, :])
        )
        speed = smooth_abs(velocity)
        return 0.5 * velocity * (h_left + h_right) - 0.5 * speed * (h_right - h_left)

    def advection_momentum(self, u, v, omega, U_m, V_m, k_energy, p, h_tot_ugrid, h_tot_vgrid,
                           dx_p_ref=None, dy_p_ref=None, taux=None, tauy=None, h_wind=None, wind_strength=None):
        """
        Advection RHS for momentum (u, v)
        """
        _dx_p_ref = dx_p_ref if dx_p_ref is not None else self.dx_p_ref
        _dy_p_ref = dy_p_ref if dy_p_ref is not None else self.dy_p_ref

        # Vortex-force + Coriolis
        if self.mom_adv_scheme in ('upwind5', 'linear_upwind5', 'linear5'):
            omega_Vm, omega_Um = self._omega_adv_upwind5(omega, U_m, V_m)
        elif self.mom_adv_scheme == 'upwind3':
            omega_Vm, omega_Um = self._omega_adv_upwind3(omega, U_m, V_m)
        else:
            omega_Vm = self.w_flux_y(omega[...,1:-1,:], V_m)
            omega_Um = self.w_flux_x(omega[...,1:-1], U_m)

        dt_u = omega_Vm + self.fstar_ugrid[...,1:-1,:] * V_m
        dt_v = -(omega_Um + self.fstar_vgrid[...,1:-1] * U_m)

        # grad pressure + k_energy
        ke_pressure = k_energy + p
        dt_u -= jnp.diff(ke_pressure, axis=-2) + _dx_p_ref
        dt_v -= jnp.diff(ke_pressure, axis=-1) + _dy_p_ref

        # wind forcing and bottom drag
        dt_u, dt_v = self.add_wind_forcing(dt_u, dt_v, taux=taux, tauy=tauy, h_wind=h_wind, wind_strength=wind_strength)
        dt_u, dt_v = self.add_bottom_drag(dt_u, dt_v, u, v)
        dt_u, dt_v = self.add_diffusion(dt_u, dt_v, u, v)

        return jnp.pad(dt_u, ((0,0), (0,0), (1, 1), (0, 0)))*self.masks.u, \
               jnp.pad(dt_v, ((0,0), (0,0), (0, 0), (1, 1)))*self.masks.v

    def _omega_adv_upwind5(self, omega, U_m, V_m):
        """
        5th-order fixed-linear upwind face reconstruction of vorticity.

        Uses the same conservative velocity-biased flux machinery as WENO, but
        replaces nonlinear WENO weights by fixed linear stencils for a smoother
        adjoint. Existing masks downgrade to 2-/4-point stencils near coasts.
        """
        omega_Vm = flux(
            omega[..., 1:-1, :], V_m,
            dim=-1,
            n_points=6,
            rec_func_2=linear2_centered,
            rec_func_4=linear4_left,
            rec_func_6=linear6_left,
            mask_2=self.masks.u_sten_wy_eq2[..., 1:-1, :],
            mask_4=self.masks.u_sten_wy_eq4[..., 1:-1, :],
            mask_6=self.masks.u_sten_wy_gt6[..., 1:-1, :])
        omega_Um = flux(
            omega[..., 1:-1], U_m,
            dim=-2,
            n_points=6,
            rec_func_2=linear2_centered,
            rec_func_4=linear4_left,
            rec_func_6=linear6_left,
            mask_2=self.masks.v_sten_wx_eq2[..., 1:-1],
            mask_4=self.masks.v_sten_wx_eq4[..., 1:-1],
            mask_6=self.masks.v_sten_wx_gt6[..., 1:-1])
        return omega_Vm, omega_Um

    def _omega_adv_upwind3(self, omega, U_m, V_m):
        """
        3rd-order upwind face reconstruction of vorticity for the vortex-force term.
        Replaces w_flux_y / w_flux_x when mom_adv_scheme='upwind3'.

        omega  shape (1, nl, nx+1, ny+1)  — on corners
        U_m    shape (1, nl, nx,   ny-1)  — interp_TP(U), u at v-grid interior
        V_m    shape (1, nl, nx-1, ny  )  — interp_TP(V), v at u-grid interior

        Returns (omega_Vm, omega_Um) with the same shapes as the WENO version:
          omega_Vm  (1, nl, nx-1, ny)
          omega_Um  (1, nl, nx,   ny-1)

        Reconstruction at face j+1/2 between omega[j] and omega[j+1]:
          vel > 0: omega_face = (-omega[j-1] + 5*omega[j]   + 2*omega[j+1]) / 6
          vel < 0: omega_face = ( 2*omega[j] + 5*omega[j+1] -   omega[j+2]) / 6
        Interior faces only (1-cell ring zeroed at boundaries).
        """
        # ---- y-direction (omega_Vm): omega[...,1:-1,:] on (nx-1, ny+1) ----
        omega_w = omega[..., 1:-1, :]   # (1, nl, nx-1, ny+1)
        Vp = jnp.where(V_m > 0.,  V_m, 0.)
        Vn = jnp.where(V_m <= 0., V_m, 0.)
        omega_Vm = jnp.zeros_like(V_m)
        # Interior y-faces: j=1,...,ny-2  (index 1:-1 in V_m, size ny)
        omega_Vm = omega_Vm.at[..., 1:-1].set(
            Vp[..., 1:-1] * (-omega_w[..., :-3] + 5.*omega_w[..., 1:-2] + 2.*omega_w[..., 2:-1]) / 6.
          + Vn[..., 1:-1] * ( 2.*omega_w[..., 1:-2] + 5.*omega_w[..., 2:-1] - omega_w[..., 3:]) / 6.
        )

        # ---- x-direction (omega_Um): omega[...,1:-1] on (nx+1, ny-1) ----
        omega_w2 = omega[..., 1:-1]     # (1, nl, nx+1, ny-1)
        Up = jnp.where(U_m > 0.,  U_m, 0.)
        Un = jnp.where(U_m <= 0., U_m, 0.)
        omega_Um = jnp.zeros_like(U_m)
        # Interior x-faces: i=1,...,nx-2  (index 1:-1 in U_m dim=-2, size nx)
        omega_Um = omega_Um.at[..., 1:-1, :].set(
            Up[..., 1:-1, :] * (-omega_w2[..., :-3, :] + 5.*omega_w2[..., 1:-2, :] + 2.*omega_w2[..., 2:-1, :]) / 6.
          + Un[..., 1:-1, :] * ( 2.*omega_w2[..., 1:-2, :] + 5.*omega_w2[..., 2:-1, :] - omega_w2[..., 3:, :]) / 6.
        )

        return omega_Vm, omega_Um

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

            # Padded metrics matching u_pad (nx+1, ny+2) and v_pad (nx+2, ny+1).
            # For Cartesian grids dx_ugrid/dy_vgrid are 0-d scalars: skip padding.
            if self.dx_ugrid.ndim >= 2:
                dx_u_ypad = jnp.pad(self.dx_ugrid, ((0,0), (1,1)), mode='edge')
                dy_v_xpad = jnp.pad(self.dy_vgrid, ((1,1), (0,0)), mode='edge')
                dx_u_int = self.dx_ugrid[1:-1, :]
                dy_u_int = self.dy_ugrid[1:-1, :]
                dx_v_int = self.dx_vgrid[:, 1:-1]
                dy_v_int = self.dy_vgrid[:, 1:-1]
            else:
                dx_u_ypad = self.dx_ugrid
                dy_v_xpad = self.dy_vgrid
                dx_u_int = self.dx_ugrid
                dy_u_int = self.dy_ugrid
                dx_v_int = self.dx_vgrid
                dy_v_int = self.dy_vgrid

            # u_phys = u / dx on padded grid
            u_phys = u_pad / dx_u_ypad
            v_phys = v_pad / dy_v_xpad

            # Laplacian at interior u-points (x: 1:-1, y: all via padding)
            lap_u = (u_phys[..., 2:, 1:-1] - 2*u_phys[..., 1:-1, 1:-1] + u_phys[..., :-2, 1:-1]) / dx_u_int**2 \
                  + (u_phys[..., 1:-1, 2:] - 2*u_phys[..., 1:-1, 1:-1] + u_phys[..., 1:-1, :-2]) / dy_u_int**2

            # Laplacian at interior v-points (y: 1:-1, x: all via padding)
            lap_v = (v_phys[..., 2:, 1:-1] - 2*v_phys[..., 1:-1, 1:-1] + v_phys[..., :-2, 1:-1]) / dx_v_int**2 \
                  + (v_phys[..., 1:-1, 2:] - 2*v_phys[..., 1:-1, 1:-1] + v_phys[..., 1:-1, :-2]) / dy_v_int**2

            # Convert back to scaled variables
            du = du + self.visc_coef * lap_u * dx_u_int
            dv = dv + self.visc_coef * lap_v * dy_v_int

        return du, dv

    def add_h_diffusion(self, h):
        """
        Laplacian diffusion κ∇²(h_phys) for layer thickness.
        Returns diffusion tendency in scaled form (h = h_phys * area).
        Critical for adjoint stability: counteracts anti-diffusivity of
        the adjoint WENO scheme.
        """
        if self.diff_coef is not None and self.diff_coef > 0:
            # Operate on the physical variable h_phys = h / area so that
            # the Laplacian is correct on non-uniform grids.  The previous
            # version applied ∇² to the scaled variable h = h_phys*area,
            # which introduces spurious terms proportional to ∇(area).
            h_phys = h / self.area
            h_phys_pad = jnp.pad(h_phys, ((0,0), (0,0), (1,1), (1,1)), mode='edge')
            lap_h_phys = (h_phys_pad[..., 2:, 1:-1] - 2*h_phys_pad[..., 1:-1, 1:-1] + h_phys_pad[..., :-2, 1:-1]) / self.dx**2 \
                       + (h_phys_pad[..., 1:-1, 2:] - 2*h_phys_pad[..., 1:-1, 1:-1] + h_phys_pad[..., 1:-1, :-2]) / self.dy**2
            return self.diff_coef * lap_h_phys * self.area * self.masks.h
        return jnp.zeros_like(h)

    def add_wind_forcing(self, du, dv, taux=None, tauy=None, h_wind=None, wind_strength=None):
        """
        Add wind forcing to the derivatives du, dv.
        taux/tauy: wind stress in Pa (N/m²) on (nx-1, ny) and (nx, ny-1) grids.
        If None, falls back to self.taux / self.tauy.
        h_wind: effective mixed-layer depth for wind-stress denominator.
                Can be a scalar or a 2D array (nx, ny) on h-grid.
                If None, falls back to self.h_wind.

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
        # Three cases:
        #   h_wind argument →  use the passed value (scalar or 2D array on h-grid).
        #                      Enables JAX AD differentiation through h_wind.
        #   self.h_wind set →  use the prescribed physical mixed-layer depth (scalar).
        #                      Required for 1-layer QG/SW models where the model's
        #                      equivalent depth H = c²/g ≈ 0.4–1 m, while the real
        #                      mixed-layer driving momentum exchange is ~50–200 m.
        #   both None       →  use self.h_ref_ugrid (model reference thickness, correct
        #                      for multi-layer models where H is the true layer depth).
        _h_wind = h_wind if h_wind is not None else self.h_wind
        if _h_wind is not None:
            _h_wind = jnp.asarray(_h_wind, dtype=self.dtype)
            if _h_wind.ndim >= 2:
                # 2D field (nx, ny) on h-grid → interpolate to interior u/v grids
                H_ref_u = jnp.maximum(0.5 * (_h_wind[:-1, :] + _h_wind[1:, :]), self.h_min)
                H_ref_v = jnp.maximum(0.5 * (_h_wind[:, :-1] + _h_wind[:, 1:]), self.h_min)
            else:
                # scalar
                H_ref_u = jnp.maximum(_h_wind, self.h_min)
                H_ref_v = jnp.maximum(_h_wind, self.h_min)
        else:
            # self.h_ref_ugrid shape: (nl, nx+1, ny) for spatially-varying H,
            #                         (nl, 1, 1)     for uniform H.
            # Take top layer (index 0), trim interior only when dim > 1.
            H0_u = self.h_ref_ugrid[0]   # (nx+1, ny) or (1, 1)
            H0_v = self.h_ref_vgrid[0]   # (nx, ny+1) or (1, 1)
            if H0_u.ndim >= 2 and H0_u.shape[0] > 1:
                H0_u = H0_u[1:-1, :]     # (nx-1, ny)
            if H0_v.ndim >= 2 and H0_v.shape[1] > 1:
                H0_v = H0_v[:, 1:-1]     # (nx, ny-1)
            # area_ugrid/vgrid may be 0-d scalars when dx/dy are uniform scalars
            # (e.g. QG mode).  Skip slicing in that case — scalars broadcast fine.
            _area_u_int = (self.area_ugrid[1:-1, :]
                           if self.area_ugrid.ndim >= 2 and self.area_ugrid.shape[0] > 1
                           else self.area_ugrid)
            _area_v_int = (self.area_vgrid[:, 1:-1]
                           if self.area_vgrid.ndim >= 2 and self.area_vgrid.shape[1] > 1
                           else self.area_vgrid)
            H_ref_u = jnp.maximum(H0_u / _area_u_int, self.h_min)
            H_ref_v = jnp.maximum(H0_v / _area_v_int, self.h_min)

        # dx/dy metrics on interior u/v faces — may be 0-d scalars for uniform grids
        _dx_u_int = (self.dx_ugrid[1:-1, :]
                     if self.dx_ugrid.ndim >= 2 and self.dx_ugrid.shape[0] > 1
                     else self.dx_ugrid)
        _dy_v_int = (self.dy_vgrid[:, 1:-1]
                     if self.dy_vgrid.ndim >= 2 and self.dy_vgrid.shape[1] > 1
                     else self.dy_vgrid)

        # Wind tendency: jnp.where so land points never compute tau/H (avoids inf*0=NaN)
        wind_u = jnp.where(
            mask_u[..., 0, :, :] > 0.5,
            _taux / (self.rho_water * H_ref_u) * _dx_u_int,
            jnp.zeros_like(du[..., 0, :, :]))
        wind_v = jnp.where(
            mask_v[..., 0, :, :] > 0.5,
            _tauy / (self.rho_water * H_ref_v) * _dy_v_int,
            jnp.zeros_like(dv[..., 0, :, :]))

        if wind_strength is not None:
            _wind_strength = jnp.clip(wind_strength, -1.0, 2.0)
            if _wind_strength.ndim >= 2:
                # 2D field (nx, ny) on h-grid → interpolate to interior u/v grids
                ws_u = 0.5 * (_wind_strength[:-1, :] + _wind_strength[1:, :])  # (nx-1, ny)
                ws_v = 0.5 * (_wind_strength[:, :-1] + _wind_strength[:, 1:])  # (nx, ny-1)
            else:
                ws_u = _wind_strength
                ws_v = _wind_strength
            wind_u = (1.0 + ws_u) * wind_u
            wind_v = (1.0 + ws_v) * wind_v

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
        U = u / self.dx_ugrid**2
        V = v / self.dy_vgrid**2
        U_m = self.interp_TP(U)
        V_m = self.interp_TP(V)
        k_energy = self.comp_ke(u, U, v, V) * self.masks.h
        h_ = replicate_pad(h, self.masks.h)
        h_ugrid = 0.5 * (h_[...,1:,1:-1] + h_[...,:-1,1:-1])
        h_vgrid = 0.5 * (h_[...,1:-1,1:] + h_[...,1:-1,:-1])
        # Clamp in PHYSICAL units (metres): see advection_h note. h_tot_*grid stays
        # area-scaled for downstream consumers, so we divide by area, clamp, multiply back.
        h_tot_ugrid = smooth_clamp((_h_ref_ugrid + h_ugrid) / self.area_ugrid, self.h_min, self.h_min_sharpness) * self.area_ugrid
        h_tot_vgrid = smooth_clamp((_h_ref_vgrid + h_vgrid) / self.area_vgrid, self.h_min, self.h_min_sharpness) * self.area_vgrid

        return omega, eta, p, U, V, U_m, V_m, k_energy, h_tot_ugrid, h_tot_vgrid

    def filter_barotropic_waves(self, dt_u, dt_v, dt_h, u, v, h_tot_ugrid, h_tot_vgrid):
        """
        Inspired from https://doi.org/10.1029/2000JC900089.
        """
        # compute RHS
        u_star = (u + self.dt*dt_u) / self.dx_ugrid
        v_star = (v + self.dt*dt_v) / self.dy_vgrid
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

    def compute_time_derivatives(self, u, v, h, ref_vals=None, taux=None, tauy=None, h_wind=None, wind_strength=None, **kwargs):
        """
        Computes the state variables derivatives dt_u, dt_v, dt_h.
        ref_vals: optional tuple (h_ref, h_ref_ugrid, h_ref_vgrid, dx_p_ref, dy_p_ref)
                  for pure-functional usage (needed for correct JAX AD through H).
        taux/tauy: wind stress (overrides self.taux / self.tauy if provided).
        h_wind: effective mixed-layer depth for wind-stress (scalar or 2D on h-grid).
        Extra keyword arguments are accepted for subclasses (for example QG uses
        h_b for boundary-aware projection) and ignored by the SW dynamics.
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
            dx_p_ref, dy_p_ref, taux=taux, tauy=tauy, h_wind=h_wind, wind_strength=wind_strength)
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
        h_wind=None,
        wind_strength=None,
    ):
        """
        Performs nstep time-integration with RK3-SSP scheme.
        Memory-efficient for reverse-mode differentiation.
        taux/tauy: wind stress arrays (nx-1, ny) and (nx, ny-1).
        If None, falls back to self.taux / self.tauy set at initialisation.
        Passing them here allows time-varying wind forcing.
        wind_strength: 2D array (nx, ny) or scalar multiplier for wind forcing (default None = no scaling).
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
        if self.H_min is not None:
            H_total = jnp.maximum(H_total, self.H_min)
        if self.H_max is not None:
            H_total = jnp.minimum(H_total, self.H_max)
        ref_vals = self._compute_ref_values(H_total)
        _h_ref = ref_vals[0]  # h_ref for h_floor

        # Resolve wind stress: prefer argument, fall back to self
        _taux = taux if taux is not None else self.taux
        _tauy = tauy if tauy is not None else self.tauy

        # Resolve h_wind: combine base (self.h_wind) with perturbation
        if h_wind is not None:
            if self.h_wind is not None:
                _h_wind = self.h_wind + h_wind
            else:
                _h_wind = h_wind
        else:
            _h_wind = None  # falls back to self.h_wind inside add_wind_forcing

        # ----------------------------
        # Single time step (scheme chosen at construction time → specialised by JIT)
        # ----------------------------
        def single_step(carry, _):
            u, v, h = carry

            # Sponge as Rayleigh damping integrated within RK substages.
            # Rate γ = sponge_coef / dt  so that ∫₀ᵈᵗ γ ds ≈ sponge_coef.
            _gamma_u = (self.sponge_coef / self.dt) * self.sponge_u
            _gamma_v = (self.sponge_coef / self.dt) * self.sponge_v
            _gamma_h = (self.sponge_coef / self.dt) * self.sponge_h

            def _f(u_, v_, h_):
                """Tendency + sponge at current state."""
                dtu, dtv, dth = self.compute_time_derivatives(
                    u_, v_, h_, ref_vals, taux=_taux, tauy=_tauy,
                    h_wind=_h_wind, wind_strength=wind_strength,
                    h_b=_h_b if h_b is not None else None)
                dtu = dtu + _gamma_u * (_u_b - u_)
                dtv = dtv + _gamma_v * (_v_b - v_)
                dth = dth + _gamma_h * (_h_b - h_)
                return dtu, dtv, dth

            if self.time_scheme == 'rk2':
                # ---- Explicit midpoint (matches Qgm rk2) ----
                dt0_u, dt0_v, dt0_h = _f(u, v, h)
                u_mid = u + (self.dt * 0.5) * dt0_u
                v_mid = v + (self.dt * 0.5) * dt0_v
                h_mid = h + (self.dt * 0.5) * dt0_h
                dt1_u, dt1_v, dt1_h = _f(u_mid, v_mid, h_mid)
                u = u + self.dt * dt1_u
                v = v + self.dt * dt1_v
                h = h + self.dt * dt1_h

            elif self.time_scheme == 'rk2_ssp':
                # ---- Heun / SSP-RK2 ----
                dt0_u, dt0_v, dt0_h = _f(u, v, h)
                u1 = u + self.dt * dt0_u
                v1 = v + self.dt * dt0_v
                h1 = h + self.dt * dt0_h
                dt1_u, dt1_v, dt1_h = _f(u1, v1, h1)
                u = u + (self.dt * 0.5) * (dt0_u + dt1_u)
                v = v + (self.dt * 0.5) * (dt0_v + dt1_v)
                h = h + (self.dt * 0.5) * (dt0_h + dt1_h)

            else:
                # ---- RK3-SSP (Shu-Osher) — default ----
                dt0_u, dt0_v, dt0_h = _f(u, v, h)
                u = u + self.dt * dt0_u
                v = v + self.dt * dt0_v
                h = h + self.dt * dt0_h

                dt1_u, dt1_v, dt1_h = _f(u, v, h)
                u = u + (self.dt / 4.0) * (dt1_u - 3.0 * dt0_u)
                v = v + (self.dt / 4.0) * (dt1_v - 3.0 * dt0_v)
                h = h + (self.dt / 4.0) * (dt1_h - 3.0 * dt0_h)

                dt2_u, dt2_v, dt2_h = _f(u, v, h)
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
                # Clamp in PHYSICAL units; result stays area-scaled for the
                # mass-consistent forcing ratio u/h_tot * Fh.
                h_tot_u = smooth_clamp((ref_vals[1] + h_ugrid) / self.area_ugrid,
                                       self.h_min,
                                       self.h_min_sharpness) * self.area_ugrid
                h_tot_v = smooth_clamp((ref_vals[2] + h_vgrid) / self.area_vgrid,
                                       self.h_min,
                                       self.h_min_sharpness) * self.area_vgrid
                Fh_ = replicate_pad(_Fh, self.masks.h)
                Fh_u = 0.5 * (Fh_[..., 1:, 1:-1] + Fh_[..., :-1, 1:-1])
                Fh_v = 0.5 * (Fh_[..., 1:-1, 1:] + Fh_[..., 1:-1, :-1])
                u = u + self.dt * (-u / h_tot_u * Fh_u)
                v = v + self.dt * (-v / h_tot_v * Fh_v)
            
            u = u + self.dt * _Fu
            v = v + self.dt * _Fv
            h = h + self.dt * _Fh

            return (u, v, h), None

        # --------------------------------------
        # Checkpoint the step (full mode: checkpoint entire single_step;
        # minimises scan residuals — each backward step re-executes single_step)
        # Set self.scan_checkpoint = False before first JIT call to disable.
        # --------------------------------------
        scan_body = jax.checkpoint(single_step) if self.scan_checkpoint else single_step

        # --------------------------------------
        # Use scan instead of fori_loop
        # --------------------------------------
        if nstep > 0:
            (u, v, h), _ = lax.scan(
                scan_body,
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

    def step_tgl(self, u0, v0, h0, du0, dv0, dh0, H=None, dH=None, nstep=1, taux=None, tauy=None, h_wind=None, dh_wind=None):
        """
        Tangent Linear Model: computes the linearized evolution of perturbations.
        taux/tauy: wind stress passed through to step().
        h_wind/dh_wind: mixed-layer depth perturbation and its tangent.
        """
        def wrapped_step(x):
            u0, v0, h0, H, h_wind = x
            return self.step(u0, v0, h0, H, nstep=nstep, taux=taux, tauy=tauy, h_wind=h_wind)

        primals = ((u0, v0, h0, H, h_wind),)
        tangents = ((du0, dv0, dh0, dH, dh_wind),)

        y, dy = jax.jvp(wrapped_step, primals, tangents)

        return dy  # returns (du, dv, dh)

    def step_adj(self, u0, v0, h0, wuT, wvT, whT, H=None, nstep=1, taux=None, tauy=None, h_wind=None):
        """
        Adjoint Model: computes the adjoint propagation backward.
        taux/tauy: wind stress passed through to step().
        h_wind: mixed-layer depth perturbation (differentiable).
        """
        def wrapped_step(x):
            u0, v0, h0, H, h_wind = x
            return self.step(u0, v0, h0, H, nstep=nstep, taux=taux, tauy=tauy, h_wind=h_wind)
        primals = ((u0, v0, h0, H, h_wind),)
        cotangents = (wuT, wvT, whT)  

        y, vjp_fn = jax.vjp(wrapped_step, *primals)
        adjoints = vjp_fn(cotangents)
        return adjoints  # returns (adj_u0, adj_v0, adj_h0, adj_H, adj_h_wind)

    def step_with_tracer(
        self,
        u0,
        v0,
        h0,
        c0,
        H=None,
        nstep=1,
        u_b=None,
        v_b=None,
        h_b=None,
        c_b=None,
        Fu=None,
        Fv=None,
        Fh=None,
        Fc=None,
        taux=None,
        tauy=None,
        h_wind=None,
        wind_strength=None,
    ):
        """
        Performs nstep time-integration for (u, v, h) *and* passive tracer c.

        c0: (1, n_trac, nx, ny)  physical tracer values (e.g. °C, PSU).
        c_b: boundary value for tracer, same shape as c0 (or None → no nudging).
        Fc: external tracer forcing per time step (same shape as c0, or None).

        Tracer c is stored internally in area-scaled form (c_area = c_phys * area)
        consistent with the h convention.  The surface velocity (layer 0) is used
        for advection — the physical h-grid-interpolated velocities derived from
        the staggered (u, v).  Diagnostics (U, V) are shared between the
        momentum and tracer tendencies within each RK3 stage to avoid redundant
        computation.

        Returns (u_phys, v_phys, h_phys, c_phys).
        """

        # Prepare (u, v, h) in scaled form
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

        # Prepare tracer in area-scaled form (mirrors h = h_phys * area)
        c = jnp.asarray(c0, dtype=self.dtype) * self.masks.h * self.area  # (1, n_trac, nx, ny)
        _c_b = jnp.asarray(c_b, dtype=self.dtype) * self.masks.h * self.area if c_b is not None else c
        _Fc  = jnp.asarray(Fc, dtype=self.dtype) * self.area if Fc is not None else jnp.zeros_like(c)

        # Ref values
        H_total = self.H + H if H is not None else self.H
        if self.H_min is not None:
            H_total = jnp.maximum(H_total, self.H_min)
        if self.H_max is not None:
            H_total = jnp.minimum(H_total, self.H_max)
        ref_vals = self._compute_ref_values(H_total)

        _taux = taux if taux is not None else self.taux
        _tauy = tauy if tauy is not None else self.tauy

        if h_wind is not None:
            _h_wind = (self.h_wind + h_wind) if self.h_wind is not None else h_wind
        else:
            _h_wind = None

        def _stage_tendencies(u, v, h, c):
            """Compute RHS for (u, v, h, c) sharing diagnostics (U, V).

            Mirrors compute_time_derivatives() exactly for (u, v, h), then
            reuses the already-computed U, V for tracer advection to avoid a
            second diagnostic call per RK3 stage.
            """
            omega, eta, p, U, V, U_m, V_m, k_energy, h_tot_ugrid, h_tot_vgrid = \
                self.compute_diagnostic_variables(u, v, h, ref_vals[1], ref_vals[2])
            dt_h = self.advection_h(U, V, h, ref_vals[0]) + self.add_h_diffusion(h)
            dt_u, dt_v = self.advection_momentum(
                u, v, omega, U_m, V_m, k_energy, p,
                h_tot_ugrid, h_tot_vgrid,
                ref_vals[3], ref_vals[4],
                taux=_taux, tauy=_tauy, h_wind=_h_wind, wind_strength=wind_strength)
            if self.barotropic_filter:
                dt_u, dt_v, dt_h = self.filter_barotropic_waves(
                    dt_u, dt_v, dt_h, u, v, h_tot_ugrid, h_tot_vgrid)
            # Tracer: reuse U, V from above — no extra diagnostic call
            dt_c = self.advection_tracer(U, V, c) + self.add_tracer_diffusion(c)
            return dt_u, dt_v, dt_h, dt_c

        def single_step(carry, _):
            u, v, h, c = carry

            _gamma_u = (self.sponge_coef / self.dt) * self.sponge_u
            _gamma_v = (self.sponge_coef / self.dt) * self.sponge_v
            _gamma_h = (self.sponge_coef / self.dt) * self.sponge_h
            _gamma_c = (self.sponge_coef / self.dt) * self.sponge_h  # tracer on h-grid

            def _f(u_, v_, h_, c_):
                """Tendency + sponge at current state."""
                dtu, dtv, dth, dtc = _stage_tendencies(u_, v_, h_, c_)
                dtu = dtu + _gamma_u * (_u_b - u_)
                dtv = dtv + _gamma_v * (_v_b - v_)
                dth = dth + _gamma_h * (_h_b - h_)
                dtc = dtc + _gamma_c * (_c_b - c_)
                return dtu, dtv, dth, dtc

            if self.time_scheme == 'rk2':
                # ---- Explicit midpoint ----
                dt0_u, dt0_v, dt0_h, dt0_c = _f(u, v, h, c)
                u_m = u + (self.dt * 0.5) * dt0_u
                v_m = v + (self.dt * 0.5) * dt0_v
                h_m = h + (self.dt * 0.5) * dt0_h
                c_m = c + (self.dt * 0.5) * dt0_c
                dt1_u, dt1_v, dt1_h, dt1_c = _f(u_m, v_m, h_m, c_m)
                u = u + self.dt * dt1_u
                v = v + self.dt * dt1_v
                h = h + self.dt * dt1_h
                c = c + self.dt * dt1_c

            elif self.time_scheme == 'rk2_ssp':
                # ---- Heun / SSP-RK2 ----
                dt0_u, dt0_v, dt0_h, dt0_c = _f(u, v, h, c)
                u1 = u + self.dt * dt0_u
                v1 = v + self.dt * dt0_v
                h1 = h + self.dt * dt0_h
                c1 = c + self.dt * dt0_c
                dt1_u, dt1_v, dt1_h, dt1_c = _f(u1, v1, h1, c1)
                u = u + (self.dt * 0.5) * (dt0_u + dt1_u)
                v = v + (self.dt * 0.5) * (dt0_v + dt1_v)
                h = h + (self.dt * 0.5) * (dt0_h + dt1_h)
                c = c + (self.dt * 0.5) * (dt0_c + dt1_c)

            else:
                # ---- RK3-SSP stage 0 ----
                dt0_u, dt0_v, dt0_h, dt0_c = _f(u, v, h, c)
                u = u + self.dt * dt0_u
                v = v + self.dt * dt0_v
                h = h + self.dt * dt0_h
                c = c + self.dt * dt0_c

                # ---- RK3-SSP stage 1 ----
                dt1_u, dt1_v, dt1_h, dt1_c = _f(u, v, h, c)
                u = u + (self.dt / 4.0) * (dt1_u - 3.0 * dt0_u)
                v = v + (self.dt / 4.0) * (dt1_v - 3.0 * dt0_v)
                h = h + (self.dt / 4.0) * (dt1_h - 3.0 * dt0_h)
                c = c + (self.dt / 4.0) * (dt1_c - 3.0 * dt0_c)

                # ---- RK3-SSP stage 2 ----
                dt2_u, dt2_v, dt2_h, dt2_c = _f(u, v, h, c)
                u = u + (self.dt / 12.0) * (8.0 * dt2_u - dt1_u - dt0_u)
                v = v + (self.dt / 12.0) * (8.0 * dt2_v - dt1_v - dt0_v)
                h = h + (self.dt / 12.0) * (8.0 * dt2_h - dt1_h - dt0_h)
                c = c + (self.dt / 12.0) * (8.0 * dt2_c - dt1_c - dt0_c)

            # ---- External forcing (after RK stages) ----
            if self.forcing_momentum == 'mass_consistent':
                h_ = replicate_pad(h, self.masks.h)
                h_ugrid = 0.5 * (h_[..., 1:, 1:-1] + h_[..., :-1, 1:-1])
                h_vgrid = 0.5 * (h_[..., 1:-1, 1:] + h_[..., 1:-1, :-1])
                # Clamp in PHYSICAL units; result stays area-scaled.
                h_tot_u = smooth_clamp((ref_vals[1] + h_ugrid) / self.area_ugrid,
                                       self.h_min, self.h_min_sharpness) * self.area_ugrid
                h_tot_v = smooth_clamp((ref_vals[2] + h_vgrid) / self.area_vgrid,
                                       self.h_min, self.h_min_sharpness) * self.area_vgrid
                Fh_ = replicate_pad(_Fh, self.masks.h)
                Fh_u = 0.5 * (Fh_[..., 1:, 1:-1] + Fh_[..., :-1, 1:-1])
                Fh_v = 0.5 * (Fh_[..., 1:-1, 1:] + Fh_[..., 1:-1, :-1])
                u = u + self.dt * (-u / h_tot_u * Fh_u)
                v = v + self.dt * (-v / h_tot_v * Fh_v)

            u = u + self.dt * _Fu
            v = v + self.dt * _Fv
            h = h + self.dt * _Fh
            c = c + self.dt * _Fc

            return (u, v, h, c), None

        single_step = jax.checkpoint(single_step)

        if nstep > 0:
            (u, v, h, c), _ = lax.scan(
                single_step,
                (u, v, h, c),
                None,
                length=nstep,
            )

        # Back to physical space
        u_phys, v_phys, h_phys = self.get_physical_uvh(u, v, h, numpy=False)
        c_phys = (c / self.area) * self.masks.h   # physical tracer, masked

        return u_phys, v_phys, h_phys, c_phys

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

        # Run ADJ:  ((au0, av0, ah0, aH, ah_wind),) = M* (wu, wv, wh)
        adjoints = self.step_adj(u0, v0, h0, wu, wv, wh, nstep=nstep)
        au0, av0, ah0, _aH, _ah_wind = adjoints[0]

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
