"""
Shallow-water implementation.
Louis Thiry, Nov 2023 for IFREMER.
"""
import numpy as np
import jax.numpy as jnp 
import jax
from jax import jit
jax.config.update("jax_enable_x64", True)

from finite_diff import interp_TP, interp_TP_inv, comp_ke, div_nofluxbc
from flux import flux
from helmholtz import HelmholtzNeumannSolver
#from helmholtz_multigrid import MG_Helmholtz
from masks import Masks
from reconstruction import linear2_centered, wenoz4_left, wenoz6_left
from tools import avg_pool2d

from jax import lax

from functools import partial

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

        # gravity
        self.g_prime = param['g_prime']
        self.g = self.g_prime[0]

        # external top-layer forcing
        taux, tauy = param['taux'], param['tauy']
        self.set_wind_forcing(taux, tauy)
        self.bottom_drag_coef = param['bottom_drag_coef']

        # time
        self.dt = param['dt']
        print(f'  - integration time step {self.dt:.3e}')

        # ensemble
        self.n_ens = param['n_ens'] if 'n_ens' in param.keys() else 1

        # topography and ref values
        self.h_ref = self.H * self.area
        self.eta_ref = -self.H.sum(axis=-3) + reverse_cumsum(self.H, dim=-3)
        self.p_ref = jnp.cumsum(self.g_prime * self.eta_ref, axis=-3)
        if self.h_ref.shape[-2] != 1 and self.h_ref.shape[-1] != 1:
            #h_ref_ugrid = F.pad(self.h_ref, (0,0,1,1), mode='replicate')
            h_ref_ugrid = jnp.pad(self.h_ref, ((1, 1), (0, 0)), mode='edge')
            self.h_ref_ugrid = 0.5 * (h_ref_ugrid[...,1:,:] + h_ref_ugrid[...,:-1,:])
            #h_ref_vgrid = F.pad(self.h_ref, (1,1), mode='replicate')
            h_ref_vgrid = jnp.pad(self.h_ref, ((0, 0), (1, 1)), mode='edge')
            self.h_ref_vgrid = 0.5 * (h_ref_vgrid[...,1:] + h_ref_vgrid[...,:-1])
            self.dx_p_ref = jnp.diff(self.p_ref, axis=-2)
            self.dy_p_ref = jnp.diff(self.p_ref, axis=-1)
        else:
            self.h_ref_ugrid = self.h_ref
            self.h_ref_vgrid = self.h_ref
            self.dx_p_ref = 0.
            self.dy_p_ref = 0.

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
                if param['barotropic_filter_spectral']:
                    print('spectral approximation')
                    self.barotropic_filter_spectral = True
                    self.H_tot = self.H.sum(dim=-3, keepdim=True)
                    self.lambd = 1. / (self.g * self.dt * self.tau * self.H_tot)
                    self.helm_solver = HelmholtzNeumannSolver(
                            self.nx, self.ny, self.dx, self.dy, self.lambd,
                            self.dtype, mask=self.masks.h[0,0])
                else:
                    self.barotropic_filter_spectral = False
                    print('in exact form')
                    coef_ugrid = (self.h_tot_ugrid * self.masks.u)[0,0]
                    coef_vgrid = (self.h_tot_vgrid * self.masks.v)[0,0]
                    lambd = 1. / (self.g * self.dt * self.tau)
                    self.helm_solver = MG_Helmholtz(self.dx, self.dy,
                            self.nx, self.ny, coef_ugrid, coef_vgrid=coef_vgrid,
                            lambd=lambd, dtype=self.dtype,
                            mask=self.masks.h[0,0], niter_bottom=20,
                            use_compilation=False)
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
            self.step = jit(self.step, static_argnames=['nstep','return_ssh'])
            self.step_tgl = jit(self.step_tgl, static_argnames=['nstep','return_ssh'])
            self.step_adj = jit(self.step_adj, static_argnames=['nstep','return_ssh'])

        else:
            print('  - No compilation')


    def set_wind_forcing(self, taux, tauy):
        nx, ny = self.nx, self.ny
        assert type(taux) == float or taux.shape == (nx-1, ny), \
               f'taux must be a float or a {(nx-1, ny)} Tensor'
        assert type(tauy) == float or tauy.shape == (nx, ny-1), \
               f'taux must be a float or a {(nx-1, ny)} Tensor'
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
        #assert jnp.all(u_ * self.masks.u == u_), \
        #    'Input velocity u incoherent with domain mask, velocity must be zero out of domain.'
        #assert jnp.all(v_ * self.masks.v == v_), \
        #    'Input velocity v incoherent with domain mask, velocity must be zero out of domain.'
        # Ensure variables are masked instead of asserting
        u_ = u_ * self.masks.u
        v_ = v_ * self.masks.v
        u = u_.astype(self.dtype) * self.masks.u * self.dx
        v = v_.astype(self.dtype) * self.masks.v * self.dy
        h = h_.astype(self.dtype) * self.masks.h * self.area

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

    def advection_h(self, U, V, h):
        """
        Advection RHS for thickness perturbation h
        dt_h = - div(h_tot [u v]),  h_tot = h_ref + h
        """
        h_tot = self.h_ref + h
        h_tot_flux_y = self.h_flux_y(h_tot, V[...,1:-1])
        h_tot_flux_x = self.h_flux_x(h_tot, U[...,1:-1,:])
        return -div_nofluxbc(h_tot_flux_x, h_tot_flux_y) * self.masks.h

    def advection_momentum(self, u, v, omega, U_m, V_m, k_energy, p, h_tot_ugrid, h_tot_vgrid):
        """
        Advection RHS for momentum (u, v)
        """
        # Vortex-force + Coriolis
        omega_Vm = self.w_flux_y(omega[...,1:-1,:], V_m)
        omega_Um = self.w_flux_x(omega[...,1:-1], U_m)

        dt_u = omega_Vm + self.fstar_ugrid[...,1:-1,:] * V_m
        dt_v = -(omega_Um + self.fstar_vgrid[...,1:-1] * U_m)

        # grad pressure + k_energy
        ke_pressure = k_energy + p
        dt_u -= jnp.diff(ke_pressure, axis=-2) + self.dx_p_ref
        dt_v -= jnp.diff(ke_pressure, axis=-1) + self.dy_p_ref

        # wind forcing and bottom drag
        dt_u, dt_v = self.add_wind_forcing(dt_u, dt_v, h_tot_ugrid, h_tot_vgrid)
        dt_u, dt_v = self.add_bottom_drag(dt_u, dt_v, u, v)

        return jnp.pad(dt_u, ((0,0), (0,0), (1, 1), (0, 0)))*self.masks.u, \
               jnp.pad(dt_v, ((0,0), (0,0), (0, 0), (1, 1)))*self.masks.v

    def add_wind_forcing(self, du, dv, h_tot_ugrid, h_tot_vgrid):
        """
        Add wind forcing to the derivatives du, dv.
        """
        H_ugrid = (h_tot_ugrid) / self.area
        H_vgrid = (h_tot_vgrid) / self.area
        du = du.at[..., 0,:,:].set(du[..., 0,:,:] + self.taux / H_ugrid[...,0,1:-1,:] * self.dx)
        dv = dv.at[..., 0,:,:].set(dv[..., 0,:,:] + self.tauy / H_vgrid[...,0,:,1:-1] * self.dy)
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
        v_ = jnp.pad(v, ((0, 0), (0, 0), (1, 1), (0, 0)))#F.pad(v, (0,0,1,1))
        dx_v = jnp.diff(v_, axis=-2)
        dy_u = jnp.diff(u_, axis=-1)
        curl_uv = dx_v - dy_u
        alpha = 2 * (1 - self.slip_coef)
        omega = self.masks.w_valid * curl_uv \
              + self.masks.w_cornerout_bound * (1 - self.slip_coef) * curl_uv \
              + self.masks.w_vertical_bound * alpha * dx_v \
              - self.masks.w_horizontal_bound * alpha * dy_u
        return omega

    def compute_diagnostic_variables(self, u, v , h):
        """
        Compute the model's diagnostic variables given the prognostic
        variables self.u, self.v, self.h .
        """
        omega = self.compute_omega(u, v)
        eta = reverse_cumsum(h / self.area, dim=-3)
        p = jnp.cumsum(self.g_prime * eta, axis=-3)
        U = u / self.dx**2
        V = v / self.dy**2
        U_m = self.interp_TP(U)
        V_m = self.interp_TP(V)
        k_energy = self.comp_ke(u, U, v, V) * self.masks.h
        # self.pv = (self.interp_TP(self.omega) + self.fstar_hgrid) \
                  # / (self.h_ref + self.h)

        h_ = replicate_pad(h, self.masks.h)
        h_ugrid = 0.5 * (h_[...,1:,1:-1] + h_[...,:-1,1:-1])
        h_vgrid = 0.5 * (h_[...,1:-1,1:] + h_[...,1:-1,:-1])
        h_tot_ugrid = self.h_ref_ugrid + h_ugrid
        h_tot_vgrid = self.h_ref_vgrid + h_vgrid

        return omega, eta, p, U, V, U_m, V_m, k_energy, h_tot_ugrid, h_tot_vgrid

    def filter_barotropic_waves(self, dt_u, dt_v, dt_h, u, v, h_tot_ugrid, h_tot_vgrid):
        """
        Inspired from https://doi.org/10.1029/2000JC900089.
        """
        # compute RHS
        u_star = (u + self.dt*dt_u) / self.dx
        v_star = (v + self.dt*dt_v) / self.dy
        u_bar_star = (u_star * h_tot_ugrid).sum(axis=-3, keepdims=True) \
                     / h_tot_ugrid.sum(axis=-3, keepdim=True)
        v_bar_star = (v_star * h_tot_vgrid).sum(dim=-3, keepdims=True) \
                     / self.h_tot_vgrid.sum(axis=-3, keepdims=True)
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

    def compute_time_derivatives(self, u, v , h):
        """
        Computes the state variables derivatives dt_u, dt_v, dt_h
        """
        omega, eta, p, U, V, U_m, V_m, k_energy, h_tot_ugrid, h_tot_vgrid = \
            self.compute_diagnostic_variables(u, v , h)
        dt_h = self.advection_h(U, V, h)
        dt_u, dt_v = self.advection_momentum(u, v, omega, U_m, V_m, k_energy, p, h_tot_ugrid, h_tot_vgrid)
        if self.barotropic_filter:
            dt_u, dt_v, dt_h = self.filter_barotropic_waves(dt_u, dt_v, dt_h, u, v, h_tot_ugrid, h_tot_vgrid)

        return dt_u, dt_v, dt_h

    def h2ssh(self, h):

        A_inv = jnp.linalg.inv(self.A)  # shape: (m, l)
        E = h / (self.H)   # shape: (..., l, x, y)
        p_i = jnp.einsum('ml,...lxy->...mxy', A_inv, E)

        return p_i/9.81  # shape: (..., x, y)

    def ssh2h(self, ssh):
        """
        Computes h (..., l, x, y) from ssh (..., m, x, y).
        """
        A = self.A                   # (m, l)
        H = self.H                   # (l,)
        p_i = ssh * 9.81              # (..., m, x, y)
        
        # Compute E = A @ p_i
        E = jnp.einsum('lm,...mxy->...lxy', A, p_i)   # (..., l, x, y)
        
        h = E * H   # broadcast H over x, y: H[l] * E[..., l, x, y]
        
        return h                   # (..., l, x, y)

    def step(self, u0, v0, h0, nstep=1, return_ssh=True):
        """
        Performs one step time-integration with RK3-SSP scheme.
        """

        u, v, h = self.set_input_uvh(u0, v0, h0)

        def single_step(i, carry):
            
            u, v, h = carry
            
            # Compute time derivatives
            dt0_u, dt0_v, dt0_h = self.compute_time_derivatives(u, v , h)
            u += self.dt * dt0_u
            v += self.dt * dt0_v
            h += self.dt * dt0_h

            dt1_u, dt1_v, dt1_h = self.compute_time_derivatives(u, v , h)
            u += (self.dt/4) * (dt1_u - 3*dt0_u)
            v += (self.dt/4) * (dt1_v - 3*dt0_v)
            h += (self.dt/4) * (dt1_h - 3*dt0_h)

            dt2_u, dt2_v, dt2_h = self.compute_time_derivatives(u, v , h)
            u += (self.dt/12) * (8*dt2_u - dt1_u - dt0_u)
            v += (self.dt/12) * (8*dt2_v - dt1_v - dt0_v)
            h += (self.dt/12) * (8*dt2_h - dt1_h - dt0_h)

            return (u, v, h)
        
        # Perform nstep iterations
        u, v, h = lax.fori_loop(0, nstep, single_step, (u, v, h))

        # Back to physics
        u_phys, v_phys, h_phys = self.get_physical_uvh(u, v, h, numpy=False)

        # Update SSH
        if return_ssh:
            ssh = self.h2ssh(h_phys)
            return u_phys, v_phys, h_phys, ssh
        else:
            return u_phys, v_phys, h_phys
    
    def step_tgl(self, u0, v0, h0, du0, dv0, dh0, nstep=1, return_ssh=True):
        """
        Tangent Linear Model: computes the linearized evolution of perturbations.
        """
        def wrapped_step(x):
            u0, v0, h0 = x
            return self.step(u0, v0, h0, nstep=nstep, return_ssh=return_ssh)
        
        primals = ((u0, v0, h0),)
        tangents = ((du0, dv0, dh0),)
        
        y, dy = jax.jvp(wrapped_step, primals, tangents)
        return dy  # returns (du, dv, dh)
    
    def step_adj(self, u0, v0, h0, wuT, wvT, whT, wsshT=None, nstep=1, return_ssh=True):
        """
        Adjoint Model: computes the adjoint propagation backward.
        """
        def wrapped_step(x):
            u0, v0, h0 = x
            return self.step(u0, v0, h0, nstep=nstep, return_ssh=return_ssh)
        
        primals = ((u0, v0, h0),)
        if return_ssh:
            cotangents = (wuT, wvT, whT, wsshT)
        else:
            cotangents = (wuT, wvT, whT)
        
        y, vjp_fn = jax.vjp(wrapped_step, *primals)
        adjoints = vjp_fn(cotangents)
        return adjoints  # returns (adj_u0, adj_v0, adj_h0)

class _SW:
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

        # gravity
        self.g_prime = param['g_prime']
        self.g = self.g_prime[0]

        # external top-layer forcing
        taux, tauy = param['taux'], param['tauy']
        self.set_wind_forcing(taux, tauy)
        self.bottom_drag_coef = param['bottom_drag_coef']

        # time
        self.dt = param['dt']
        print(f'  - integration time step {self.dt:.3e}')

        # ensemble
        self.n_ens = param['n_ens'] if 'n_ens' in param.keys() else 1

        # topography and ref values
        self.h_ref = self.H * self.area
        self.eta_ref = -self.H.sum(axis=-3) + reverse_cumsum(self.H, dim=-3)
        self.p_ref = jnp.cumsum(self.g_prime * self.eta_ref, axis=-3)
        if self.h_ref.shape[-2] != 1 and self.h_ref.shape[-1] != 1:
            #h_ref_ugrid = F.pad(self.h_ref, (0,0,1,1), mode='replicate')
            h_ref_ugrid = jnp.pad(self.h_ref, ((1, 1), (0, 0)), mode='edge')
            self.h_ref_ugrid = 0.5 * (h_ref_ugrid[...,1:,:] + h_ref_ugrid[...,:-1,:])
            #h_ref_vgrid = F.pad(self.h_ref, (1,1), mode='replicate')
            h_ref_vgrid = jnp.pad(self.h_ref, ((0, 0), (1, 1)), mode='edge')
            self.h_ref_vgrid = 0.5 * (h_ref_vgrid[...,1:] + h_ref_vgrid[...,:-1])
            self.dx_p_ref = jnp.diff(self.p_ref, axis=-2)
            self.dy_p_ref = jnp.diff(self.p_ref, axis=-1)
        else:
            self.h_ref_ugrid = self.h_ref
            self.h_ref_vgrid = self.h_ref
            self.dx_p_ref = 0.
            self.dy_p_ref = 0.


        # initialize variables
        base_shape = (self.n_ens, self.nl,)
        self.h = jnp.zeros(base_shape + (self.nx, self.ny), **self.arr_kwargs)
        self.u = jnp.zeros(base_shape + (self.nx+1, self.ny), **self.arr_kwargs)
        self.v = jnp.zeros(base_shape + (self.nx, self.ny+1), **self.arr_kwargs)
        self.comp_ke = comp_ke
        self.interp_TP = interp_TP
        self.compute_diagnostic_variables()

        # utils and flux computation functions
        self.comp_ke = comp_ke
        self.interp_TP = interp_TP
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
                if param['barotropic_filter_spectral']:
                    print('spectral approximation')
                    self.barotropic_filter_spectral = True
                    self.H_tot = self.H.sum(dim=-3, keepdim=True)
                    self.lambd = 1. / (self.g * self.dt * self.tau * self.H_tot)
                    self.helm_solver = HelmholtzNeumannSolver(
                            self.nx, self.ny, self.dx, self.dy, self.lambd,
                            self.dtype, mask=self.masks.h[0,0])
                else:
                    self.barotropic_filter_spectral = False
                    print('in exact form')
                    coef_ugrid = (self.h_tot_ugrid * self.masks.u)[0,0]
                    coef_vgrid = (self.h_tot_vgrid * self.masks.v)[0,0]
                    lambd = 1. / (self.g * self.dt * self.tau)
                    self.helm_solver = MG_Helmholtz(self.dx, self.dy,
                            self.nx, self.ny, coef_ugrid, coef_vgrid=coef_vgrid,
                            lambd=lambd, dtype=self.dtype,
                            mask=self.masks.h[0,0], niter_bottom=20,
                            use_compilation=False)
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
            self.step = jit(self.step)
        else:
            print('  - No compilation')


    def set_wind_forcing(self, taux, tauy):
        nx, ny = self.nx, self.ny
        assert type(taux) == float or taux.shape == (nx-1, ny), \
               f'taux must be a float or a {(nx-1, ny)} Tensor'
        assert type(tauy) == float or tauy.shape == (nx, ny-1), \
               f'taux must be a float or a {(nx-1, ny)} Tensor'
        self.taux = taux
        self.tauy = tauy


    def get_physical_uvh(self, numpy=False):
        """Get physical variables u_phys, v_phys, h_phys from state variables."""
        u_phys = (self.u / self.dx)
        v_phys = (self.v / self.dy)
        h_phys = (self.h / self.area)

        return (np.array(u_phys), np.array(v_phys), np.array(h_phys)) if numpy \
               else (u_phys, v_phys, h_phys)


    def set_physical_uvh(self, u_phys, v_phys, h_phy, offline=False):
        """
        Set state variables with physical variables u_phys, v_phys, h_phys.
        """
        u_ = jnp.array(u_phys) if isinstance(u_phys, np.ndarray) else u_phys
        v_ = jnp.array(v_phys) if isinstance(v_phys, np.ndarray) else v_phys
        h_ = jnp.array(h_phys) if isinstance(h_phys, np.ndarray) else h_phys
        assert u_ * self.masks.u == u_, \
                'Input velocity u incoherent with domain mask, ' \
                'velocity must be zero out of domain.'
        assert v_ * self.masks.v == v_, \
                'Input velocity v incoherent with domain mask, ' \
                'velocity must be zero out of domain.'
        u = u_.type(self.dtype) * self.masks.u * self.dx
        v = v_.type(self.dtype) * self.masks.v * self.dy
        h = h_.type(self.dtype) * self.masks.h * self.area

        if offline:
            return u, v, h
        else:
            self.u = u
            self.v = v
            self.h = h
            self.compute_diagnostic_variables()


    def get_print_info(self):
        """
        Returns a string with summary of current variables.
        """
        hl_mean = (self.h / self.area).mean((-1,-2)).squeeze()
        eta = (self.eta)
        u, v, h = self.u / self.dx, self.v / self.dy, self.h / self.area
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


    def advection_h(self):
        """
        Advection RHS for thickness perturbation h
        dt_h = - div(h_tot [u v]),  h_tot = h_ref + h
        """
        h_tot = self.h_ref + self.h
        h_tot_flux_y = self.h_flux_y(h_tot, self.V[...,1:-1])
        h_tot_flux_x = self.h_flux_x(h_tot, self.U[...,1:-1,:])
        return -div_nofluxbc(h_tot_flux_x, h_tot_flux_y) * self.masks.h


    def advection_momentum(self):
        """
        Advection RHS for momentum (u, v)
        """
        # Vortex-force + Coriolis
        omega_Vm = self.w_flux_y(self.omega[...,1:-1,:], self.V_m)
        omega_Um = self.w_flux_x(self.omega[...,1:-1], self.U_m)

        dt_u = omega_Vm + self.fstar_ugrid[...,1:-1,:] * self.V_m
        dt_v = -(omega_Um + self.fstar_vgrid[...,1:-1] * self.U_m)

        # grad pressure + k_energy
        ke_pressure = self.k_energy + self.p
        dt_u -= jnp.diff(ke_pressure, axis=-2) + self.dx_p_ref
        dt_v -= jnp.diff(ke_pressure, axis=-1) + self.dy_p_ref

        # wind forcing and bottom drag
        dt_u, dt_v = self.add_wind_forcing(dt_u, dt_v)
        dt_u, dt_v = self.add_bottom_drag(dt_u, dt_v)

        return jnp.pad(dt_u, ((0,0), (0,0), (1, 1), (0, 0)))*self.masks.u, \
               jnp.pad(dt_v, ((0,0), (0,0), (0, 0), (1, 1)))*self.masks.v


    def add_wind_forcing(self, du, dv):
        """
        Add wind forcing to the derivatives du, dv.
        """
        H_ugrid = (self.h_tot_ugrid) / self.area
        H_vgrid = (self.h_tot_vgrid) / self.area
        du = du.at[..., 0,:,:].set(du[..., 0,:,:] + self.taux / H_ugrid[...,0,1:-1,:] * self.dx)
        dv = dv.at[..., 0,:,:].set(dv[..., 0,:,:] + self.tauy / H_vgrid[...,0,:,1:-1] * self.dy)
        return du, dv


    def add_bottom_drag(self, du, dv):
        """
        Add bottom drag to the derivatives du, dv.
        """
        du = du.at[...,-1,:,:].set(du[...,-1,:,:] - self.bottom_drag_coef * self.u[...,-1,1:-1,:])
        dv = dv.at[...,-1,:,:].set(dv[...,-1,:,:] - self.bottom_drag_coef * self.v[...,-1,:,1:-1])
        return du, dv


    def compute_omega(self, u, v):
        """
        Pad u and v using boundary conditions (free-slip, partial free-slip,
        no-slip).
        """
        u_ = jnp.pad(u, ((0, 0), (0, 0), (0, 0), (1, 1)))
        v_ = jnp.pad(v, ((0, 0), (0, 0), (1, 1), (0, 0)))#F.pad(v, (0,0,1,1))
        dx_v = jnp.diff(v_, axis=-2)
        dy_u = jnp.diff(u_, axis=-1)
        curl_uv = dx_v - dy_u
        alpha = 2 * (1 - self.slip_coef)
        omega = self.masks.w_valid * curl_uv \
              + self.masks.w_cornerout_bound * (1 - self.slip_coef) * curl_uv \
              + self.masks.w_vertical_bound * alpha * dx_v \
              - self.masks.w_horizontal_bound * alpha * dy_u
        return omega


    def compute_diagnostic_variables(self, ):
        """
        Compute the model's diagnostic variables given the prognostic
        variables self.u, self.v, self.h .
        """
        self.omega = self.compute_omega(self.u, self.v)
        self.eta = reverse_cumsum(self.h / self.area, dim=-3)
        self.p = jnp.cumsum(self.g_prime * self.eta, axis=-3)
        self.U = self.u / self.dx**2
        self.V = self.v / self.dy**2
        self.U_m = self.interp_TP(self.U)
        self.V_m = self.interp_TP(self.V)
        self.k_energy = self.comp_ke(self.u, self.U, self.v, self.V) * self.masks.h
        # self.pv = (self.interp_TP(self.omega) + self.fstar_hgrid) \
                  # / (self.h_ref + self.h)

        h_ = replicate_pad(self.h, self.masks.h)
        self.h_ugrid = 0.5 * (h_[...,1:,1:-1] + h_[...,:-1,1:-1])
        self.h_vgrid = 0.5 * (h_[...,1:-1,1:] + h_[...,1:-1,:-1])
        self.h_tot_ugrid = self.h_ref_ugrid + self.h_ugrid
        self.h_tot_vgrid = self.h_ref_vgrid + self.h_vgrid


    def filter_barotropic_waves(self, dt_u, dt_v, dt_h):
        """
        Inspired from https://doi.org/10.1029/2000JC900089.
        """
        # compute RHS
        u_star = (self.u + self.dt*dt_u) / self.dx
        v_star = (self.v + self.dt*dt_v) / self.dy
        u_bar_star = (u_star * self.h_tot_ugrid).sum(dim=-3, keepdim=True) \
                     / self.h_tot_ugrid.sum(dim=-3, keepdim=True)
        v_bar_star = (v_star * self.h_tot_vgrid).sum(dim=-3, keepdim=True) \
                     / self.h_tot_vgrid.sum(dim=-3, keepdim=True)
        if self.barotropic_filter_spectral:
            rhs = 1. / (self.g * self.dt * self.tau) * (
                    torch.diff(u_bar_star, dim=-2) / self.dx \
                + torch.diff(v_bar_star, dim=-1) / self.dy)
            w_surf_imp = self.helm_solver.solve(rhs)
        else:
            rhs = 1. / (self.g * self.dt * self.tau) * (
                    torch.diff(self.h_tot_ugrid * u_bar_star, dim=-2) / self.dx \
                  + torch.diff(self.h_tot_vgrid * v_bar_star, dim=-1) / self.dy)
            coef_ugrid = (self.h_tot_ugrid * self.masks.u)[0,0]
            coef_vgrid = (self.h_tot_vgrid * self.masks.v)[0,0]
            w_surf_imp = self.helm_solver.solve(rhs, coef_ugrid, coef_vgrid)
            # WIP

        filt_u = F.pad(-self.g * self.tau * torch.diff(w_surf_imp, dim=-2), (0,0,1,1)) * self.masks.u
        filt_v = F.pad(-self.g * self.tau * torch.diff(w_surf_imp, dim=-1), (1,1)) * self.masks.v


        return dt_u + filt_u, \
               dt_v + filt_v, \
               dt_h

    def compute_time_derivatives(self):
        """
        Computes the state variables derivatives dt_u, dt_v, dt_h
        """
        self.compute_diagnostic_variables()
        dt_h = self.advection_h()
        dt_u, dt_v = self.advection_momentum()
        if self.barotropic_filter:
            dt_u, dt_v, dt_h = self.filter_barotropic_waves(dt_u, dt_v, dt_h)

        return dt_u, dt_v, dt_h

    def step(self, u0, v0, h0):
        """
        Performs one step time-integration with RK3-SSP scheme.
        """

        u,v,h = self.set_physical_uvh(u0, v0, h0, offline=True)

        dt0_u, dt0_v, dt0_h = self.compute_time_derivatives()
        u += self.dt * dt0_u
        v += self.dt * dt0_v
        h += self.dt * dt0_h

        dt1_u, dt1_v, dt1_h = self.compute_time_derivatives()
        u += (self.dt/4) * (dt1_u - 3*dt0_u)
        v += (self.dt/4) * (dt1_v - 3*dt0_v)
        h += (self.dt/4) * (dt1_h - 3*dt0_h)

        dt2_u, dt2_v, dt2_h = self.compute_time_derivatives()
        u += (self.dt/12) * (8*dt2_u - dt1_u - dt0_u)
        v += (self.dt/12) * (8*dt2_v - dt1_v - dt0_v)
        h += (self.dt/12) * (8*dt2_h - dt1_h - dt0_h)

        # Back to physics
        return self.get_physical_uvh(numpy=False)