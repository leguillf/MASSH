"""
Pytorch multilayer QG as projected SW, Louis Thiry, 9. oct. 2023.
  - QG herits from SW class, prognostic variables: u, v, h
  - DST spectral solver for QG elliptic equation
"""
import numpy as np


from helmholtz import compute_laplace_dstI, solve_helmholtz_dstI, dstI2D,\
                      solve_helmholtz_dstI_cmm, compute_capacitance_matrices
from finite_diff import grad_perp
from sw import SW, inv_reverse_cumsum

from jax import jit
from jax import numpy as jnp

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
        self.grad_perp = grad_perp


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

    def hgrid_pressure_to_wgrid(self, p_i):
        """Build a W-grid pressure from h-grid pressure without edge shrinkage."""
        pad_width = ((0, 0),) * (p_i.ndim - 2) + ((1, 0), (1, 0))
        return jnp.pad(p_i, pad_width, mode='edge')

    def _compute_qg_background(self, h_b):
        """Derive background pressure and interior PV from area-scaled h_b."""
        ssh_b = h_b / self.area
        pb_i = self.g_prime.astype(self.dtype) * ssh_b
        pb = self.hgrid_pressure_to_wgrid(pb_i)
        u_b, v_b, h_b_G = self.G(pb, p_i=pb_i)
        qb = self.Q(u_b, v_b, h_b_G)
        return pb, pb_i, qb

    def add_wind_forcing(self, du, dv, **kwargs):
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


    def QoG_inv(self, elliptic_rhs, pb=None, pb_i=None, qb=None):
        """(Q o G)^{-1}: Helmholtz solve with optional background correction.

        Mirrors Qgm.pv2h(q, hb, qb):
          qin          = elliptic_rhs - qb        (background subtraction)
          p_interior   = Helmholtz_solve(qin)     (homogeneous W-grid BC)
          p_full       = zeros
          p_full[interior] = p_interior
          p_full      += pb                       (restore background)

        Parameters
        ----------
        elliptic_rhs : (1, nl, nx-1, ny-1)  interior PV from Q operator
        pb           : (1, nl, nx+1, ny+1)  background W-grid pressure, or None
        qb           : (1, nl, nx-1, ny-1)  background interior PV, or None
        """
        # Background subtraction (mirrors Qgm: qin = q[interior] - qb[interior])
        helmholtz_rhs_input = elliptic_rhs - qb if qb is not None else elliptic_rhs

        # Layer-to-mode transform
        helmholtz_rhs = jnp.einsum('lm,...mxy->...lxy', self.Cl2m, helmholtz_rhs_input)

        # Helmholtz solve (homogeneous Dirichlet on W-grid boundary)
        if self.cap_matrices is not None:
            p_modes = solve_helmholtz_dstI_cmm(
                helmholtz_rhs * self.masks.psi[..., 1:-1, 1:-1],
                self.helmholtz_dstI, self.cap_matrices,
                self.masks.psi_irrbound_xids,
                self.masks.psi_irrbound_yids,
                self.masks.psi)
        else:
            p_modes = solve_helmholtz_dstI(helmholtz_rhs, self.helmholtz_dstI)

        # Mass correction only when no background (free-surface uniqueness)
        if qb is None:
            alpha = -p_modes.mean((-1, -2), keepdims=True) / self.homsol_wgrid_mean
            p_modes = p_modes + alpha * self.homsol_wgrid

        # Mode-to-layer: full W-grid (1, nl, nx+1, ny+1)
        # (solve_helmholtz_dstI already pads interior solution to full W-grid)
        p_wgrid = jnp.einsum('lm,...mxy->...lxy', self.Cm2l, p_modes)

        # Add background on full W-grid (mirrors Qgm: h+=hb)
        p_qg = (pb + p_wgrid) if pb is not None else p_wgrid

        p_qg_i = pb_i + self.interp_TP(p_wgrid) if pb_i is not None else self.interp_TP(p_qg)
        return p_qg, p_qg_i

    def Q(self, u, v, h):
        """Q operator: compute elliptic equation r.h.s."""
        f0, H, area = self.f0, self.H, self.area
        omega = jnp.diff(v[...,1:-1], axis=-2) - jnp.diff(u[...,1:-1,:], axis=-1)
        elliptic_rhs_interior = (omega - f0 * self.interp_TP(h) / H) * (f0 / area)

        # For normal operation, return interior-only elliptic RHS
        # Boundary conditions are handled in the QoG_inv method via background subtraction
        return elliptic_rhs_interior

    def project_qg(self, u, v, h, pb=None, pb_i=None, qb=None):
        """ QG projector P = G o (Q o G)^{-1} o Q """
        return self.G(*self.QoG_inv(self.Q(u, v, h), pb=pb, pb_i=pb_i, qb=qb))

    def step(self, *args, **kwargs):
        # Extract h_b here so it is NOT forwarded into the SW scan body.
        # The scan body no longer does per-substage state projection.
        h_b = kwargs.pop('h_b', None)
        sponge_coef = self.sponge_coef
        self.sponge_coef = 0.0

        if h_b is not None:
            # Compute QG background once per step (constant within a step)
            h_b_internal = jnp.asarray(h_b, dtype=self.dtype) * self.area * self.masks.h
            pb, pb_i, qb = self._compute_qg_background(h_b_internal)
            # Pre-project initial state onto QG manifold (replaces per-substage projections)
            u0, v0, h0 = args[0], args[1], args[2]
            u_qg, v_qg, h_qg = self.set_input_uvh(u0, v0, h0)
            u_qg, v_qg, h_qg = self.project_qg(u_qg, v_qg, h_qg, pb=pb, pb_i=pb_i, qb=qb)
            u0p, v0p, h0p = self.get_physical_uvh(u_qg, v_qg, h_qg, numpy=False)
            args = (u0p, v0p, h0p) + args[3:]

        try:
            # h_b is not passed to super: scan substages don't carry it
            u_phys, v_phys, h_phys = super().step(*args, **kwargs)
        finally:
            self.sponge_coef = sponge_coef

        if h_b is None:
            return u_phys, v_phys, h_phys

        # Post-step projection using pre-computed background (no recompute)
        u, v, h = self.set_input_uvh(u_phys, v_phys, h_phys)
        u, v, h = self.project_qg(u, v, h, pb=pb, pb_i=pb_i, qb=qb)
        return self.get_physical_uvh(u, v, h, numpy=False)

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

    def compute_diagnostic_variables(self, u, v, h, h_ref_ugrid=None, h_ref_vgrid=None):
        return super().compute_diagnostic_variables(u, v, h, h_ref_ugrid, h_ref_vgrid)
    
    def compute_pv(self, omega, h):
        """Compute potential vorticity."""
        pv = self.interp_TP(omega) / self.area - self.f0 * (h / self.h_ref)
        return pv

    def compute_time_derivatives(self, u, v, h, ref_vals=None, **kwargs):
        kwargs.pop('h_b', None)  # h_b is now handled at step() level
        dt_uvh_sw = super().compute_time_derivatives(u, v, h, ref_vals, **kwargs)
        dt_uvh_qg = self.project_qg(*dt_uvh_sw)

        self.dt_h = dt_uvh_sw[2]
        self.P_dt_h = dt_uvh_qg[2]
        # P2_dt_h removed: diagnostic-only, cost 1 Helmholtz solve per substage

        self.compute_ageostrophic_velocity(dt_uvh_qg, dt_uvh_sw)

        return dt_uvh_qg

