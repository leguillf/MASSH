"""
JAX implementation of multigrid solver for 2D generalized Helmoltz equation
    ∇.(c∇u) - λu = rhs
with homegenous Neumann BC, where the coefficent c possibly varies in space.
Assuming staggered grid:
    o---v---o---v---o
    |       |       |
    u   x   u   x   u
    |       |       |
    o---v---o---v---o
    |       |       |
    u   x   u   x   u
    |       |       |
    o---v---o---v---o
  - function and rhs sampled at cell centers (x).
  - Neumann bc apply on cell edges (u and v).

Domain defined by a mask embedded in a rectangle.

Louis Thiry, 2023 (PyTorch) — converted to JAX 2024
"""
import numpy as np
import jax
import jax.numpy as jnp
from tools import avg_pool2d



def compute_nlevels(n):
    if n <= 8:
        return 1
    highest_powerof2_divisor =  np.log2(n & (~(n - 1))).astype(int)
    while n // 2**highest_powerof2_divisor < 8:
        highest_powerof2_divisor -= 1
    return 1 + highest_powerof2_divisor


def compute_mask_uvgrids(mask):
    """Computes the mask on the u and v grids given the mask on the center grid.
    Input mask: (nx, ny) or (1, nx, ny).
    Returns mask_ugrid (nx+1, ny) and mask_vgrid (nx, ny+1) as 2D arrays.
    """
    mask_ = jnp.expand_dims(mask, 0) if mask.ndim == 2 else mask
    mask_ugrid = (avg_pool2d(
            mask_, (2, 1), stride=(1, 1), padding=(1, 0), divisor_override=1) > 1.5
        ).astype(mask.dtype)
    mask_vgrid = (avg_pool2d(
            mask_, (1, 2), stride=(1, 1), padding=(0, 1), divisor_override=1) > 1.5
        ).astype(mask.dtype)
    return mask_ugrid[0], mask_vgrid[0]


def compute_helmholtz_matrix(dx, dy, lambd, mask, coef_ugrid, coef_vgrid, dtype):
    """Computes the equivalent matrix of a Helmholtz operator on a masked domain."""
    def helmholtz(f, dx, dy, lambd, mask, coef_ugrid, coef_vgrid):
        f_ = jnp.pad(f, [(0, 0)] * (f.ndim - 2) + [(1, 1), (1, 1)])
        dx_f = jnp.diff(f_[..., 1:-1], axis=-2)
        dy_f = jnp.diff(f_[..., 1:-1, :], axis=-1)
        Hf = (jnp.diff(coef_ugrid * dx_f, axis=-2) / dx**2
              + jnp.diff(coef_vgrid * dy_f, axis=-1) / dy**2
              - lambd * f)
        return Hf * mask

    i_s, j_s = jnp.where(mask)
    N = len(i_s)
    h_matrix = jnp.zeros((lambd.shape[0], N, N), dtype=dtype)
    for n in range(N):
        x = jnp.zeros((lambd.shape[0],) + mask.shape, dtype=dtype)
        x = x.at[:, i_s[n], j_s[n]].set(1)
        H_x = helmholtz(x, dx, dy, lambd, mask, coef_ugrid, coef_vgrid)
        h_matrix = h_matrix.at[:, n].set(H_x[:, i_s, j_s])
    return i_s, j_s, h_matrix


def jacobi_smoothing(f, rhs, dx, dy, mask, coef_ugrid, coef_vgrid, omega, lambd):
    """Jacobi smoothing operator on masked grid."""
    dxm2 = 1. / dx**2
    dym2 = 1. / dy**2
    cu_ip1_j = coef_ugrid[..., 1:, :]
    cu_i_j   = coef_ugrid[..., :-1, :]
    cv_i_jp1 = coef_vgrid[..., 1:]
    cv_i_j   = coef_vgrid[..., :-1]
    factor = mask / (
            lambd
          + dxm2 * (cu_ip1_j + cu_i_j)
          + dym2 * (cv_i_jp1 + cv_i_j))

    for _ in range(6):
        f_ = jnp.pad(f, [(0, 0)] * (f.ndim - 2) + [(1, 1), (1, 1)])
        f_ip1_j = f_[..., 2:,  1:-1]
        f_im1_j = f_[..., :-2, 1:-1]
        f_i_jp1 = f_[..., 1:-1, 2:]
        f_i_jm1 = f_[..., 1:-1, :-2]
        f = omega * factor * (
                  dxm2 * (cu_ip1_j * f_ip1_j + cu_i_j * f_im1_j)
                + dym2 * (cv_i_jp1 * f_i_jp1 + cv_i_j * f_i_jm1)
                - rhs) \
          + (1 - omega) * f
    return f


def residual(f, rhs, dx, dy, mask, coef_ugrid, coef_vgrid, lambd):
    """Compute 2D Helmholtz equation with Neumann BC residual:
        res = rhs - Hf,  Hf = ∇.(c∇f) - λf
    The residual is zero iff f is a solution.
    """
    f_ = jnp.pad(f, [(0, 0)] * (f.ndim - 2) + [(1, 1), (1, 1)])
    dx_f = jnp.diff(f_[..., 1:-1], axis=-2)
    dy_f = jnp.diff(f_[..., 1:-1, :], axis=-1)
    Hf = mask * (
            jnp.diff(coef_ugrid * dx_f, axis=-2) / dx**2
          + jnp.diff(coef_vgrid * dy_f, axis=-1) / dy**2
          - lambd * f)
    return rhs - Hf


def prolong(v, divisor=16):
    """Cell-centered prolongation of the 2D field v."""
    nx, ny = v.shape[-2:]
    batch_shape = v.shape[:-2]
    v_ = jnp.pad(v, [(0, 0)] * (v.ndim - 2) + [(1, 1), (1, 1)])

    # Four fine-grid quadrant values, each shape (*batch, nx, ny)
    q00 = (9 * v_[..., 1:-1, 1:-1]
           + 3 * (v_[..., 1:-1, 0:-2] + v_[..., 0:-2, 1:-1])
           +      v_[..., 0:-2, 0:-2])
    q10 = (9 * v_[..., 1:-1, 1:-1]
           + 3 * (v_[..., 1:-1, 0:-2] + v_[..., 2:,   1:-1])
           +      v_[..., 2:,   0:-2])
    q01 = (9 * v_[..., 1:-1, 1:-1]
           + 3 * (v_[..., 1:-1, 2:]   + v_[..., 0:-2, 1:-1])
           +      v_[..., 0:-2, 2:])
    q11 = (9 * v_[..., 1:-1, 1:-1]
           + 3 * (v_[..., 2:,   1:-1] + v_[..., 1:-1, 2:])
           +      v_[..., 2:,   2:])

    # Scatter into fine-grid output using non-overlapping stride-2 writes
    v_f = jnp.zeros(batch_shape + (2 * nx, 2 * ny), dtype=v.dtype)
    v_f = v_f.at[..., 0::2, 0::2].set(q00)
    v_f = v_f.at[..., 1::2, 0::2].set(q10)
    v_f = v_f.at[..., 0::2, 1::2].set(q01)
    v_f = v_f.at[..., 1::2, 1::2].set(q11)
    return v_f / divisor


def restrict(v, divisor=4):
    """Cell-centered coarse-grid restriction of the 2D field v."""
    return (  v[..., :-1:2, :-1:2]
            + v[...,  1::2, :-1:2]
            + v[...,   ::2,  1::2]
            + v[...,  1::2,  1::2]) / divisor




class MG_Helmholtz():
    """
    Multigrid solver for generalized Helmholtz equations
        ∇.(c∇u) - λu = rhs
    c being a possibly non-constant coefficient for
    masked domains embedded in a rectangle.
    """

    def __init__(self,
                 dx,
                 dy,
                 nx,
                 ny,
                 coef_ugrid,
                 coef_vgrid,
                 n_levels=None,
                 lambd=None,
                 tol=1e-8,
                 max_ite=20,
                 dtype=jnp.float64,
                 device='cpu',   # kept for API compatibility, unused in JAX
                 mask=None,
                 niter_bottom=-1,
                 use_compilation=True):

        if lambd is None:
            self.lambd = jnp.zeros((1,), dtype=dtype)
        else:
            self.lambd = jnp.asarray(lambd, dtype=dtype)

        if n_levels is None:
            n_levels = min(compute_nlevels(nx), compute_nlevels(ny))
        assert n_levels >= 1, f'at least 1 level needed, got {n_levels}'
        N = 2**(n_levels - 1)
        assert nx % N == 0 and ny % N == 0, \
               (f'invalid {n_levels=}, {nx=} and {ny=} must be divisible '
                f'by 2**(n_levels-1)={N}')

        print(f'JAX multigrid solver ∇.(c∇f) - λf = rhs, '
              f'λ={np.array(jnp.ravel(self.lambd))}, {dtype}, '
              f'n_levels={n_levels}')

        self.dtype = dtype

        # Grid parameters
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.shape = (int(self.lambd.shape[0]), self.nx, self.ny)

        # Pre-/post-smoothing relaxation factor
        self.omega_pre  = 0.95
        self.omega_post = 0.95

        # Multigrid algo parameters
        self.n_levels     = n_levels
        self.max_ite      = max_ite
        self.tol          = tol
        self.niter_bottom = niter_bottom

        # fine-to-coarse and coarse-to-fine operators (may be JIT-compiled below)
        self.restrict = restrict
        self.prolong  = prolong

        # Mask hierarchy
        self.compute_mask_hierarchy(mask)

        # Restriction and prolongation divisors
        self.compute_divisor_hierarchy()

        # Divergence coefficient hierarchy
        self.compute_coefficient_hierarchy(coef_ugrid, coef_vgrid)

        # Bottom solve: pre-compute pseudo-inverse of the Helmholtz matrix
        if self.niter_bottom <= 0:
            dx_bottom     = dx * 2**(n_levels - 1)
            dy_bottom     = dy * 2**(n_levels - 1)
            mask_bottom   = self.masks[-1][0]
            coef_u_bottom = self.coefs_ugrid[-1][0]
            coef_v_bottom = self.coefs_vgrid[-1][0]
            i_s, j_s, h_mat = compute_helmholtz_matrix(
                    dx_bottom, dy_bottom, self.lambd,
                    mask_bottom, coef_u_bottom, coef_v_bottom, dtype)
            self.i_s = i_s
            self.j_s = j_s
            self.bottom_inv_mat = jnp.linalg.pinv(h_mat)

        # JIT-compile individual operators
        if use_compilation:
            self.smooth   = jax.jit(jacobi_smoothing)
            self.residual = jax.jit(residual)
            self.restrict = jax.jit(restrict)
            self.prolong  = jax.jit(prolong)
        else:
            self.smooth   = jacobi_smoothing
            self.residual = residual
            self.restrict = restrict
            self.prolong  = prolong

    # ------------------------------------------------------------------
    # Hierarchy construction
    # ------------------------------------------------------------------

    def compute_mask_hierarchy(self, mask):
        if mask is None:
            mask = jnp.ones((self.nx, self.ny), dtype=self.dtype)
        assert mask.shape[-2] == self.nx and mask.shape[-1] == self.ny, \
                f'Invalid mask shape {mask.shape} != ({self.nx}, {self.ny})'
        mask = jnp.expand_dims(mask, 0)   # (1, nx, ny)
        mask_ugrid, mask_vgrid = compute_mask_uvgrids(mask)
        self.masks       = [mask]
        self.masks_ugrid = [mask_ugrid]
        self.masks_vgrid = [mask_vgrid]
        for _ in range(1, self.n_levels):
            mask = (self.restrict(self.masks[-1]) > 0.5).astype(self.dtype)
            mask_ugrid, mask_vgrid = compute_mask_uvgrids(mask)
            self.masks.append(mask)
            self.masks_ugrid.append(mask_ugrid)
            self.masks_vgrid.append(mask_vgrid)


    def compute_divisor_hierarchy(self):
        self.divisors_restrict = [None]
        for n in range(1, self.n_levels):
            div = restrict(self.masks[n - 1], divisor=1)
            self.divisors_restrict.append(jnp.maximum(div, jnp.ones_like(div)))
        self.divisors_prolong = []
        for n in range(self.n_levels - 1):
            div = prolong(self.masks[n + 1], divisor=1)
            self.divisors_prolong.append(jnp.maximum(div, jnp.ones_like(div)))
        self.divisors_prolong.append(None)


    def compute_coefficient_hierarchy(self, coef_ugrid, coef_vgrid):
        nx, ny = self.nx, self.ny
        assert coef_ugrid.shape[-2] == nx + 1 and coef_ugrid.shape[-1] == ny, \
               f'Invalid coef shape {coef_ugrid.shape[-2:]} != ({nx+1}, {ny})'
        assert coef_vgrid.shape[-2] == nx and coef_vgrid.shape[-1] == ny + 1, \
               f'Invalid coef shape {coef_vgrid.shape[-2:]} != ({nx}, {ny+1})'
        # Cell-center average of u/v coefficients, shape (1, nx, ny)
        coef = 0.25 * jnp.expand_dims(
            coef_ugrid[..., 1:, :] + coef_ugrid[..., :-1, :]
            + coef_vgrid[..., 1:]  + coef_vgrid[..., :-1], axis=0)
        self.coefs_ugrid = [jnp.expand_dims(coef_ugrid, 0) * self.masks_ugrid[0]]
        self.coefs_vgrid = [jnp.expand_dims(coef_vgrid, 0) * self.masks_vgrid[0]]
        for i in range(1, self.n_levels):
            coef = restrict(coef, self.divisors_restrict[i]) * self.masks[i]
            self.coefs_ugrid.append(
                avg_pool2d(coef, (2, 1), stride=(1, 1), padding=(1, 0))
                * self.masks_ugrid[i])
            self.coefs_vgrid.append(
                avg_pool2d(coef, (1, 2), stride=(1, 1), padding=(0, 1))
                * self.masks_vgrid[i])


    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------

    def solve(self, rhs, coef_ugrid=None, coef_vgrid=None):
        if coef_ugrid is not None and coef_vgrid is not None:
            self.compute_coefficient_hierarchy(coef_ugrid, coef_vgrid)
        return self.FMG_Helmholtz(rhs, self.dx, self.dy)

    def solve_smooth(self, rhs, coef_ugrid=None, coef_vgrid=None):
        if coef_ugrid is None or coef_vgrid is None:
            coef_ugrid = self.coefs_ugrid[0]
            coef_vgrid = self.coefs_vgrid[0]
        f = jnp.zeros_like(rhs)
        for _ in range(1000):
            f = self.smooth(f, rhs, self.dx, self.dy, self.masks[0],
                            coef_ugrid, coef_vgrid, self.omega_pre, self.lambd)
        return f

    def solve_V(self, rhs):
        dx, dy = self.dx, self.dy
        f = jnp.zeros_like(rhs)
        f = jax.lax.fori_loop(
            0, self.max_ite,
            lambda _, fi: self.V_cycle(fi, rhs, self.n_levels, dx, dy),
            f)
        return f


    def V_cycle(self, f, rhs, n_levels, dx, dy, level=0):
        mask       = self.masks[level]
        coef_ugrid = self.coefs_ugrid[level]
        coef_vgrid = self.coefs_vgrid[level]

        # Bottom solve
        if level == n_levels - 1:
            if self.niter_bottom <= 0:   # matrix inversion
                f = jnp.zeros_like(rhs)
                f = f.at[..., self.i_s, self.j_s].set(
                    jnp.einsum('...l,...lm->...m',
                               rhs[..., self.i_s, self.j_s],
                               self.bottom_inv_mat))
            else:                         # Jacobi smoothing
                for _ in range(self.niter_bottom):
                    f = self.smooth(f, rhs, dx, dy, mask, coef_ugrid,
                                    coef_vgrid, self.omega_pre, self.lambd)
            return f

        # Step 1: pre-smooth
        f = self.smooth(f, rhs, dx, dy, mask, coef_ugrid, coef_vgrid,
                        self.omega_pre, self.lambd)

        # Step 2: restrict residual to coarse grid
        res = self.residual(f, rhs, dx, dy, mask, coef_ugrid, coef_vgrid, self.lambd)
        res_coarse = self.restrict(res, self.divisors_restrict[level + 1]) \
                     * self.masks[level + 1]

        # Step 3: recursively solve on coarse grid
        eps_coarse = jnp.zeros_like(res_coarse)
        eps_coarse = self.V_cycle(eps_coarse, res_coarse,
                                  n_levels, dx * 2, dy * 2, level=level + 1)

        # Step 4: prolongate correction and add
        eps = self.prolong(eps_coarse, self.divisors_prolong[level]) * mask
        f   = f + eps

        # Step 5: post-smooth
        f = self.smooth(f, rhs, dx, dy, mask, coef_ugrid, coef_vgrid,
                        self.omega_post, self.lambd)
        return f


    def FMG_Helmholtz(self, rhs, dx, dy):
        """Full Multigrid cycle."""
        rhs_list = [rhs]
        for i in range(1, self.n_levels):
            rhs_list.append(
                self.restrict(rhs_list[-1], self.divisors_restrict[i])
                * self.masks[i])

        # Bottom solve
        f = jnp.zeros_like(rhs_list[-1])
        rhs_bottom = rhs_list[-1]
        if self.niter_bottom <= 0:   # matrix inversion
            f = f.at[..., self.i_s, self.j_s].set(
                jnp.einsum('...l,...lm->...m',
                           rhs_bottom[..., self.i_s, self.j_s],
                           self.bottom_inv_mat))
        else:                         # Jacobi smoothing
            k = 2**(self.n_levels - 1)
            for _ in range(self.niter_bottom):
                f = self.smooth(f, rhs_bottom, k * dx, k * dy,
                                self.masks[-1], self.coefs_ugrid[-1],
                                self.coefs_vgrid[-1], self.omega_pre, self.lambd)

        # Upsample + V-cycle to high resolution
        for i in range(2, self.n_levels + 1):
            k = 2**(self.n_levels - i)
            f = self.prolong(f, self.divisors_prolong[-i]) * self.masks[-i]
            f = self.V_cycle(f, rhs_list[-i], self.n_levels,
                             k * dx, k * dy, level=self.n_levels - i)

        # Final V-cycles: fixed number of iterations (no norm check → no all-reduce per step)
        f = jax.lax.fori_loop(
            0, self.max_ite,
            lambda _, fi: self.V_cycle(fi, rhs, self.n_levels, dx, dy),
            f)
        return f



if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.size': 16})
    plt.ion()

    def helmholtz_op(f, dx, dy, coef_ugrid, coef_vgrid, lambd):
        f_ = jnp.pad(f, [(0, 0)] * (f.ndim - 2) + [(1, 1), (1, 1)])
        dx_f = jnp.diff(f_[..., 1:-1], axis=-2) / dx
        dy_f = jnp.diff(f_[..., 1:-1, :], axis=-1) / dy
        return (jnp.diff(coef_ugrid * dx_f, axis=-2) / dx
              + jnp.diff(coef_vgrid * dy_f, axis=-1) / dy
              - lambd * f)

    dtype = jnp.float64
    key  = jax.random.PRNGKey(0)

    nx, ny   = 64, 64
    dx = jnp.array(0.1, dtype=dtype)
    dy = jnp.array(0.1, dtype=dtype)
    n_levels = None

    # Masks
    sq_mask = jnp.ones((nx, ny), dtype=dtype)
    ii, jj  = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing='ij')
    circ_mask = ((ii + 0.5 - (nx + 0.5) / 2)**2
               + (jj + 0.5 - (ny + 0.5) / 2)**2 <= (nx / 2)**2).astype(dtype)

    # Variable coefficient: bilinearly upsampled from a coarse random field
    key, subkey = jax.random.split(key)
    coef_var_lr = jax.random.uniform(subkey, (nx // 2, ny // 2),
                                     minval=0.9, maxval=1.1, dtype=dtype)
    coef_var = jax.image.resize(coef_var_lr, (nx, ny), method='linear')
    coef_cst = jnp.ones((nx, ny), dtype=dtype)

    def make_uvcoefs(c):
        c_ = jnp.pad(c, ((1, 1), (1, 1)), mode='edge')
        coef_u = 0.5 * (c_[1:,  1:-1] + c_[:-1, 1:-1])   # (nx+1, ny)
        coef_v = 0.5 * (c_[1:-1, 1:]  + c_[1:-1, :-1])   # (nx, ny+1)
        return coef_u, coef_v

    coef_cst_ugrid, coef_cst_vgrid = make_uvcoefs(coef_cst)
    coef_var_ugrid, coef_var_vgrid = make_uvcoefs(coef_var)
    lambd = jnp.ones((1, 1, 1), dtype=dtype)

    for mask, coef_u_, coef_v_, title in [
            (sq_mask,   coef_cst_ugrid, coef_cst_vgrid, 'square domain, cst coeff'),
            (sq_mask,   coef_var_ugrid, coef_var_vgrid, 'square domain, var coeff'),
            (circ_mask, coef_cst_ugrid, coef_cst_vgrid, 'masked circular domain, cst coeff'),
            (circ_mask, coef_var_ugrid, coef_var_vgrid, 'masked circular domain, var coeff'),
    ]:
        mask_ugrid, mask_vgrid = compute_mask_uvgrids(mask)
        coef_ugrid = coef_u_ * mask_ugrid
        coef_vgrid = coef_v_ * mask_vgrid

        key, subkey = jax.random.split(key)
        f = jax.random.normal(subkey, (1, nx, ny), dtype=dtype) * mask
        helm_f = helmholtz_op(f, dx, dy, coef_ugrid, coef_vgrid, lambd)

        mg = MG_Helmholtz(dx, dy, nx, ny, coef_ugrid, coef_vgrid,
                          n_levels=n_levels, lambd=lambd,
                          dtype=dtype, mask=mask, niter_bottom=8)
        f_rec    = mg.solve(helm_f)
        abs_diff = jnp.abs(f_rec - f)

        n_plots = 4
        fig, ax = plt.subplots(1, n_plots, figsize=(16, 4))
        fig.suptitle(f'Solving ∇.(c∇u) - λu = rhs, {title}')
        fig.colorbar(ax[0].imshow(np.array(coef_ugrid).T,  origin='lower'), ax=ax[0])
        ax[0].set_title('coefficient u grid')
        fig.colorbar(ax[1].imshow(np.array(coef_vgrid).T,  origin='lower'), ax=ax[1])
        ax[1].set_title('coefficient v grid')
        fig.colorbar(ax[2].imshow(np.array(f[0]).T,        origin='lower'), ax=ax[2])
        ax[2].set_title('true f')
        fig.colorbar(ax[3].imshow(np.array(abs_diff[0]).T, origin='lower'), ax=ax[3])
        ax[3].set_title('abs diff')
        [(ax[k].set_xticks([]), ax[k].set_yticks([])) for k in range(n_plots)]
        fig.tight_layout()
