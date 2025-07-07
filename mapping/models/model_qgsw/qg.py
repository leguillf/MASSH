"""
Pytorch multilayer QG as projected SW, Louis Thiry, 9. oct. 2023.
  - QG herits from SW class, prognostic variables: u, v, h
  - DST spectral solver for QG elliptic equation
"""
import numpy as np


from helmholtz import compute_laplace_dstI, solve_helmholtz_dstI, dstI2D,\
                      solve_helmholtz_dstI_cmm, compute_capacitance_matrices
from finite_diff import grad_perp
from sw import SW

from jax import jit
from jax import numpy as jnp

import matplotlib.pylab as plt 

class QG(SW):
    """Multilayer quasi-geostrophic model as projected SW."""

    def __init__(self, param):
        super().__init__(param)
        assert self.H.shape[-2:] == (1,1), \
                'H must me constant in space for ' \
                'qg approximation, i.e. have shape (...,1,1)' \
                f'got shape shape {self.H.shape}'

        # init matrices for elliptic equation
        self.compute_auxillary_matrices()

        # precompile functions
        self.grad_perp = grad_perp#torch.jit.trace(grad_perp, (self.p,))

        self.plot = True


    def compute_auxillary_matrices(self):
        # A operator
        H, g_prime = self.H.squeeze(), self.g_prime.squeeze()
        self.A = jnp.zeros((self.nl,self.nl), **self.arr_kwargs)
        if self.nl == 1:
            self.A = self.A.at[0,0].set(1./(H*g_prime))
        else:
            self.A = self.A.at[0,0].set(1./(H[0]*g_prime[0]) + 1./(H[0]*g_prime[1]))
            self.A = self.A.at[0,1].set(-1./(H[0]*g_prime[1]))
            for i in range(1, self.nl-1):
                self.A = self.A.at[i,i-1].set(-1./(H[i]*g_prime[i]))
                self.A = self.A.at[i,i].set(1./H[i]*(1/g_prime[i+1] + 1/g_prime[i]))
                self.A = self.A.at[i,i+1].set(-1./(H[i]*g_prime[i+1]))
            self.A = self.A.at[-1,-1].set(1./(H[self.nl-1]*g_prime[self.nl-1]))
            self.A = self.A.at[-1,-2].set(-1./(H[self.nl-1]*g_prime[self.nl-1]))

        # layer-to-mode and mode-to-layer matrices
        lambd_r, R = jnp.linalg.eig(self.A)
        lambd_l, L = jnp.linalg.eig(self.A.T)
        self.lambd = lambd_r.real.reshape((1, self.nl, 1, 1))
        with np.printoptions(precision=1):
            print('  - Rossby deformation Radii (km): ',
                1e-3 / np.sqrt(self.f0**2*self.lambd).squeeze())
        R, L = R.real, L.real
        self.Cl2m = jnp.diag(1./jnp.diag(L.T @ R)) @ L.T
        self.Cm2l = R

        # For Helmholtz equations
        nl, nx, ny = self.nl, self.nx, self.ny
        laplace_dstI = jnp.expand_dims(
            jnp.expand_dims(compute_laplace_dstI(
                nx, ny, self.dx, self.dy, self.arr_kwargs), axis=0),
                axis=0)
        self.helmholtz_dstI =  laplace_dstI - self.f0**2 * self.lambd

        cst_wgrid = jnp.ones((1, nl, nx+1, ny+1), **self.arr_kwargs)
        if len(self.masks.psi_irrbound_xids) > 0:
            self.cap_matrices = compute_capacitance_matrices(
                self.helmholtz_dstI, self.masks.psi_irrbound_xids,
                self.masks.psi_irrbound_yids)
            sol_wgrid = solve_helmholtz_dstI_cmm(
                    (cst_wgrid*self.masks.psi)[...,1:-1,1:-1],
                    self.helmholtz_dstI, self.cap_matrices,
                    self.masks.psi_irrbound_xids,
                    self.masks.psi_irrbound_yids,
                    self.masks.psi)
        else:
            self.cap_matrices = None
            sol_wgrid = solve_helmholtz_dstI(cst_wgrid[...,1:-1,1:-1], self.helmholtz_dstI)

        self.homsol_wgrid = cst_wgrid + sol_wgrid * self.f0**2 * self.lambd
        self.homsol_wgrid_mean = self.homsol_wgrid.mean((-1,-2), keepdims=True)
        self.homsol_hgrid = self.interp_TP(self.homsol_wgrid)
        self.homsol_hgrid_mean = self.homsol_hgrid.mean((-1,-2), keepdims=True)


    def add_wind_forcing(self, du, dv, h_tot_ugrid, h_tot_vgrid):
        du = du.at[..., 0,:,:].set(du[..., 0,:,:] + self.taux / self.H[0] * self.dx) 
        dv = dv.at[..., 0,:,:].set(dv[..., 0,:,:] + self.tauy / self.H[0] * self.dy) 
        return du, dv

    def set_physical_uvh(self, u_phys, v_phys, h_phys):
        #super().set_physical_uvh(u_phys, v_phys, h_phys)
        #super().compute_time_derivatives()
        self.u, self.v, self.h = self.project_qg(self.u, self.v, self.h)
        self.compute_diagnostic_variables()

    def G(self, p, p_i=None):

        """ G operator. """
        p_i = self.interp_TP(p) if p_i is None else p_i
        dx, dy = self.dx, self.dy

        # geostrophic balance
        u = -jnp.diff(p, axis=-1) / dy / self.f0 * dx * self.masks.u
        v = jnp.diff(p, axis=-2) / dx / self.f0 * dy  * self.masks.v

        u = jnp.where(jnp.isnan(u), 0, u)
        v = jnp.where(jnp.isnan(v), 0, v)

        u = u.at[..., :, 0].set(0)
        u = u.at[..., :, -1].set(0)
        v = v.at[..., 0, :].set(0)
        v = v.at[..., -1, :].set(0)
    
        h = self.H * jnp.einsum('lm,...mxy->...lxy', self.A, p_i) * self.area * self.masks.h
        h = jnp.where(jnp.isnan(h), 0, h)

        return u, v, h


    def QoG_inv(self, elliptic_rhs, pbc=None):
        """(Q o G)^{-1} operator: solve elliptic equation with mass conservation """

        if pbc is not None:
            # geostrophic balance
            g = 9.81
            ubc, vbc, hbc = self.G(pbc)
            qbc = self.Q(ubc, vbc, hbc, pbc=pbc)  
            self.plot = False
            elliptic_rhs = elliptic_rhs - qbc

        helmholtz_rhs = jnp.einsum('lm,...mxy->...lxy', self.Cl2m, elliptic_rhs)
        if self.cap_matrices is not None:
            p_modes = solve_helmholtz_dstI_cmm(
                    helmholtz_rhs*self.masks.psi[...,1:-1,1:-1],
                    self.helmholtz_dstI, self.cap_matrices,
                    self.masks.psi_irrbound_xids,
                    self.masks.psi_irrbound_yids,
                    self.masks.psi)
        else:
            p_modes = solve_helmholtz_dstI(helmholtz_rhs, self.helmholtz_dstI)
    
        # Add homogeneous solutions to ensure mass conservation
        if pbc is not None:
            p_modes += pbc
        else:
            alpha = -p_modes.mean((-1,-2), keepdims=True) / self.homsol_wgrid_mean
            p_modes += alpha * self.homsol_wgrid
        p_qg = jnp.einsum('lm,...mxy->...lxy', self.Cm2l, p_modes)
        p_qg_i = self.interp_TP(p_qg)

        return p_qg, p_qg_i

    def Q(self, u, v, h, pbc=None):
        """Q operator: compute elliptic equation r.h.s."""
        f0, H, area = self.f0, self.H, self.area
        omega = jnp.diff(v[...,1:-1], axis=-2) - jnp.diff(u[...,1:-1,:], axis=-1)
        elliptic_rhs = (omega - f0 * self.interp_TP(h) / H) * (f0 / area)

        # Boundary conditions
        if pbc is not None:
            g = 9.81
            elliptic_rhs = elliptic_rhs * (1 - self.masks.h_bc) - (f0/g * pbc[...,1:-1,1:-1] / H) * (f0 / area) * self.masks.h_bc
            
        return elliptic_rhs

    def project_qg(self, u, v, h, pbc=None):
        """ QG projector P = G o (Q o G)^{-1} o Q """
        return self.G(*self.QoG_inv(self.Q(u, v, h, pbc=pbc), pbc=pbc))

    def compute_ageostrophic_velocity(self, dt_uvh_qg, dt_uvh_sw):
        u_a = -(dt_uvh_qg[1] - dt_uvh_sw[1]) / self.f0 / self.dy
        v_a = (dt_uvh_qg[0] - dt_uvh_sw[0]) / self.f0 / self.dx
        k_energy_a = 0.25 * (
                u_a[...,1:]**2 + u_a[...,:-1]**2
                + v_a[...,1:,:]**2 + v_a[...,:-1,:]**2)
        omega_a = jnp.diff(v_a, axis=-2) / self.dx \
                     - jnp.diff(u_a, axis=-1) / self.dy
        div_a = jnp.diff(u_a[...,1:-1], axis=-2) / self.dx \
                   + jnp.diff(v_a[...,1:-1,:], axis=-1) / self.dy
        
        return u_a, v_a, k_energy_a, omega_a, div_a

    def compute_diagnostic_variables(self, u, v, h):
        return super().compute_diagnostic_variables(u, v, h)
    
    def compute_pv(self, omega, h):
        """Compute potential vorticity."""
        pv = self.interp_TP(omega) / self.area - self.f0 * (h / self.h_ref)
        return pv

    def compute_time_derivatives(self, u, v, h, pbc=None):
        dt_uvh_sw = super().compute_time_derivatives(u, v, h)
        dt_uvh_qg = self.project_qg(*dt_uvh_sw, pbc=pbc)

        self.dt_h = dt_uvh_sw[2]
        self.P_dt_h = dt_uvh_qg[2]
        self.P2_dt_h = self.project_qg(*dt_uvh_qg, pbc=pbc)[2]

        self.compute_ageostrophic_velocity(dt_uvh_qg, dt_uvh_sw)

        return dt_uvh_qg

