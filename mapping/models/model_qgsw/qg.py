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
        # Optional background field handling (mimicking jqgm boundary treatment)
        # When set via set_boundary_ssh, we subtract background PV during inversion
        # and add background SSH back afterward: qin = q - qb; solve; h += hb
        self._ssh_b = None         # Background SSH on T-grid (1,1,nx,ny)
        self._pb = None            # Background pressure on W-grid (1,nl,nx+1,ny+1)
        self._pb_i = None          # Background pressure on T-grid (1,nl,nx,ny)
        self._qb = None            # Background PV (1,nl,nx+1,ny+1)


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

    def set_boundary_ssh(self, ssh_b):
        """
        Define a background boundary SSH field that will be used to compute
        a background PV field. During inversion, we subtract this background PV
        from the interior, solve with homogeneous boundaries, then add the SSH back.
        This mimics the jqgm.pv2h approach: qin = q - qb, solve, h += hb.

        Parameters
        ssh_b: array-like (nx, ny) or (1, nl, nx, ny)
            Background SSH field used as boundary condition.
        """
        # Normalize input to T-grid shape (1,1,nx,ny)
        ssh_b = jnp.array(ssh_b, dtype=self.dtype)
        if ssh_b.ndim == 2:
            if ssh_b.shape == (self.nx, self.ny):
                ssh_T = ssh_b[None, None, ...]
            elif ssh_b.shape == (self.nx+1, self.ny+1):
                ssh_T = self.interp_TP(ssh_b[None, None, ...])
            else:
                raise ValueError(f"set_boundary_ssh: unexpected 2D shape {ssh_b.shape}")
        elif ssh_b.ndim == 3:
            if ssh_b.shape[-2:] == (self.nx, self.ny):
                ssh_T = ssh_b[None, ...]  # -> (1,1,nx,ny)
            elif ssh_b.shape[-2:] == (self.nx+1, self.ny+1):
                ssh_T = self.interp_TP(ssh_b[None, ...])
            else:
                raise ValueError(f"set_boundary_ssh: unexpected 3D shape {ssh_b.shape}")
        elif ssh_b.ndim == 4:
            if ssh_b.shape[-2:] == (self.nx, self.ny):
                ssh_T = ssh_b
            elif ssh_b.shape[-2:] == (self.nx+1, self.ny+1):
                ssh_T = self.interp_TP(ssh_b)
            else:
                raise ValueError(f"set_boundary_ssh: unexpected 4D shape {ssh_b.shape}")
        else:
            raise ValueError(f"set_boundary_ssh: unsupported ndim={ssh_b.ndim}")

        # Clean NaNs and apply ocean mask
        ssh_T = jnp.where(jnp.isfinite(ssh_T), ssh_T, 0.0) * self.masks.h

        # Convert SSH to pressure on T-grid: p_i = g' * ssh_T  
        # g_prime (nl,1,1) x ssh_T (1,1,nx,ny) => (1,nl,nx,ny)
        pb_i = (self.g_prime.astype(self.dtype)) * ssh_T
        
        # Map to W-grid for reconstruction
        pb = self.interp_TP_inv(pb_i) #* self.masks.psi

        # Compute the full background u,v,h fields
        ub, vb, hb = self.G(pb, p_i=pb_i)
        
        # Compute the background PV field WITHOUT boundary conditions
        # (to avoid recursion during setup)
        f0, H, area = self.f0, self.H, self.area
        omega = jnp.diff(vb[...,1:-1], axis=-2) - jnp.diff(ub[...,1:-1,:], axis=-1)
        qb_interior = (omega[0,0] - f0 * ssh_b / H.squeeze()) * (f0 / area)
        plt.figure()
        plt.pcolormesh(qb_interior)
        plt.show()
        
        # Create full W-grid qb with boundary conditions applied
        qb = jnp.zeros_like(self.masks.psi)  # Full W-grid shape
        qb = qb.at[..., 1:-1, 1:-1].set(qb_interior)  # Fill interior
        
        # Apply boundary PV following jqgm: q[ind12] = -g*f/c^2 * hb[ind12]
        # This is BEFORE the (f0/area) scaling applied in the Q operator
        # Apply boundary PV following Q function formula: (- f0 * ssh_b / H) * (f0 / area)
        boundary_mask = self.masks.not_psi[0, 0, :, :].astype(bool)  # (nx+1, ny+1)
        ssh_b_W = self.interp_TP_inv(ssh_T)[0, 0, :, :]  # (nx+1, ny+1)
        boundary_pv = (- f0 * ssh_b_W / self.H[0, 0, 0]) * (f0 / self.area)
        
        #qb = qb.at[0, 0, :, :].set(
        #    jnp.where(boundary_mask, boundary_pv, qb[0, 0, :, :])
        #)
        
        # Zero out land points
        #land_mask = ~self.masks.psi[0, 0, :, :].astype(bool)
        #qb = qb.at[0, 0, :, :].set(
        #    jnp.where(land_mask, 0.0, qb[0, 0, :, :])
        #)

        # Cache the background fields
        self._ssh_b = ssh_T  # Background SSH on T-grid (1,1,nx,ny)
        self._pb = pb        # Background pressure on W-grid (1,nl,nx+1,ny+1)
        self._pb_i = pb_i    # Background pressure on T-grid (1,nl,nx,ny)
        self._qb = qb        # Background PV (1,nl,nx+1,ny+1)


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
        #u = u.at[..., :, -1].set(0)
        v = v.at[..., 0, :].set(0)
        #v = v.at[..., -1, :].set(0)
    
        h = self.H * jnp.einsum('lm,...mxy->...lxy', self.A, p_i) * self.area * self.masks.h
        h = jnp.where(jnp.isnan(h), 0, h)

        return u, v, h


    def QoG_inv(self, elliptic_rhs):
        """(Q o G)^{-1} operator: solve elliptic equation following jqgm.py approach.
        
        Following jqgm.pv2h exactly: 
        1. Create full W-grid PV with boundary conditions applied
        2. Subtract background PV from interior: qin = q[interior] - qb[interior]
        3. Solve elliptic equation with homogeneous boundaries
        4. Add background SSH back: h += hb
        """
        # If boundary conditions are set, follow jqgm approach exactly
        if hasattr(self, '_qb') and self._qb is not None:
            # elliptic_rhs is (nl, nx-1, ny-1), need to create full W-grid PV 
            # First ensure we have the right shape for the interior assignment
            if elliptic_rhs.ndim == 3:
                # Shape is (nl, nx-1, ny-1), need (1, nl, nx-1, ny-1)
                elliptic_rhs = elliptic_rhs[None, ...]
            
            # Create full W-grid PV with boundary conditions
            full_elliptic_rhs = jnp.zeros_like(self.masks.psi)  # Shape: (1, nl, nx+1, ny+1)
            full_elliptic_rhs = full_elliptic_rhs.at[..., 1:-1, 1:-1].set(elliptic_rhs)
            
            # Apply boundary conditions from background PV at boundary points
            # Following jqgm: q[ind12] = -g*f/c^2 * hb[ind12] (before f0/area scaling)
            boundary_mask = self.masks.not_psi[0, 0, :, :].astype(bool)
            ssh_b_W = self.interp_TP_inv(self._ssh_b)[0, 0, :, :]
            f0 = self.f0
            boundary_pv = (- f0 * ssh_b_W / self.H.squeeze()) * (f0 / self.area) #-g_prime_f0 * ssh_b_W
            
            full_elliptic_rhs = full_elliptic_rhs.at[0, 0, :, :].set(
                jnp.where(boundary_mask, boundary_pv, full_elliptic_rhs[0, 0, :, :])
            )
            
            # Zero out land points
            land_mask = ~self.masks.psi[0, 0, :, :].astype(bool)
            full_elliptic_rhs = full_elliptic_rhs.at[0, 0, :, :].set(
                jnp.where(land_mask, 0.0, full_elliptic_rhs[0, 0, :, :])
            )
            
            # Subtract background PV from full field (jqgm: qin = q - qb)
            elliptic_rhs_corrected = full_elliptic_rhs - self._qb
            # Extract interior for helmholtz solver
            helmholtz_rhs_input = elliptic_rhs_corrected[..., 1:-1, 1:-1]
        else:
            # No boundary conditions: elliptic_rhs is already interior-only
            helmholtz_rhs_input = elliptic_rhs
        
        # Convert elliptic RHS to modal space
        helmholtz_rhs = jnp.einsum('lm,...mxy->...lxy', self.Cl2m, helmholtz_rhs_input)
        
        # Solve Helmholtz equation (with homogeneous boundaries due to background subtraction)
        if self.cap_matrices is not None:
            p_modes = solve_helmholtz_dstI_cmm(
                    helmholtz_rhs*self.masks.psi[...,1:-1,1:-1],
                    self.helmholtz_dstI, self.cap_matrices,
                    self.masks.psi_irrbound_xids,
                    self.masks.psi_irrbound_yids,
                    self.masks.psi)
        else:
            p_modes = solve_helmholtz_dstI(helmholtz_rhs, self.helmholtz_dstI)

        # Apply homogeneous correction for mass conservation only if no background
        if not hasattr(self, '_qb') or self._qb is None:
            alpha = -p_modes.mean((-1,-2), keepdims=True) / self.homsol_wgrid_mean
            p_modes += alpha * self.homsol_wgrid

        # Convert back to physical space
        p_qg = jnp.einsum('lm,...mxy->...lxy', self.Cm2l, p_modes)
        
        # Add background SSH back (jqgm: h += hb)
        if hasattr(self, '_pb') and self._pb is not None:
            p_qg = p_qg + self._pb
            
        # Apply mask and interpolate to T-grid
        #p_qg = p_qg * self.masks.psi
        p_qg_i = self.interp_TP(p_qg)

        return p_qg, p_qg_i

    def Q(self, u, v, h):
        """Q operator: compute elliptic equation r.h.s."""
        f0, H, area = self.f0, self.H, self.area
        omega = jnp.diff(v[...,1:-1], axis=-2) - jnp.diff(u[...,1:-1,:], axis=-1)
        elliptic_rhs_interior = (omega - f0 * self.interp_TP(h) / H) * (f0 / area)

        # For normal operation, return interior-only elliptic RHS
        # Boundary conditions are handled in the QoG_inv method via background subtraction
        return elliptic_rhs_interior

    def project_qg(self, u, v, h):
        """ QG projector P = G o (Q o G)^{-1} o Q """
        return self.G(*self.QoG_inv(self.Q(u, v, h)))

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

    def compute_time_derivatives(self, u, v, h):
        dt_uvh_sw = super().compute_time_derivatives(u, v, h)
        dt_uvh_qg = self.project_qg(*dt_uvh_sw)

        self.dt_h = dt_uvh_sw[2]
        self.P_dt_h = dt_uvh_qg[2]
        self.P2_dt_h = self.project_qg(*dt_uvh_qg)[2]

        self.compute_ageostrophic_velocity(dt_uvh_qg, dt_uvh_sw)

        return dt_uvh_qg

