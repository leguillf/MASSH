"""
Pytorch implementation of multigrid solver for 2D generalized Helmoltz equation
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

Louis Thiry, 2023
"""
import numpy as np



def compute_nlevels(n):
    if n <= 8:
        return 1
    highest_powerof2_divisor =  np.log2(n & (~(n - 1))).astype(int)
    while n // 2**highest_powerof2_divisor < 8:
        highest_powerof2_divisor -= 1
    return 1 + highest_powerof2_divisor


def compute_mask_uvgrids(mask):
    """Computes the mask on the u and v grids given the mask on the
    center grid."""
    mask_ = mask.unsqueeze(0) if len(mask.shape) == 2 else mask
    mask_ugrid = (F.avg_pool2d(
            mask_, (2,1), stride=(1,1), padding=(1,0)) > 3/4
        ).type(mask.dtype)
    mask_vgrid = (F.avg_pool2d(
            mask_, (1,2), stride=(1,1), padding=(0,1)) > 3/4
        ).type(mask.dtype)
    return (mask_ugrid[0], mask_vgrid[0]) \
           if len(mask.shape) == 2 \
           else (mask_ugrid[0], mask_vgrid[0])


def compute_helmholtz_matrix(dx, dy, lambd, mask, coef_ugrid,
                             coef_vgrid, dtype, device):
    """Computes the equivalent matrix of a Helmholtz operator
    on a masked domain."""
    def helmholtz(f, dx, dy, lambd, mask, coef_ugrid, coef_vgrid):
        f_ = F.pad(f, (1,1,1,1))
        dx_f = torch.diff(f_[...,1:-1], dim=-2)
        dy_f = torch.diff(f_[...,1:-1,:], dim=-1)
        Hf =  torch.diff(coef_ugrid*dx_f, dim=-2) / dx**2 \
             + torch.diff(coef_vgrid*dy_f, dim=-1) / dy**2 \
             - lambd * f
        return Hf * mask

    i_s, j_s = torch.where(mask)
    N = len(i_s)
    h_matrix = torch.zeros((lambd.shape[0], N, N), dtype=dtype, device=device)
    for n in range(N):
        x = torch.zeros((lambd.shape[0],)+mask.shape, dtype=dtype, device=device)
        x[:,i_s[n],j_s[n]] = 1
        H_x = helmholtz(x, dx, dy, lambd, mask, coef_ugrid, coef_vgrid)
        h_matrix[:,n] = H_x[:,i_s, j_s]
    return i_s, j_s, h_matrix


def jacobi_smoothing(f, rhs, dx, dy, mask, coef_ugrid,
                     coef_vgrid, omega, lambd):
    """Jacobi smoothing operator on masked grid."""

    dxm2 = 1. / dx**2
    dym2 = 1. / dy**2
    cu_ip1_j = coef_ugrid[...,1:,:]
    cu_i_j = coef_ugrid[...,:-1,:]
    cv_i_jp1 = coef_vgrid[...,1:]
    cv_i_j = coef_vgrid[...,:-1]
    factor = mask / (
            lambd \
          + dxm2 * (cu_ip1_j + cu_i_j) \
          + dym2 * (cv_i_jp1 + cv_i_j)
        )

    for i in range(6):
        f_ = F.pad(f, (1,1,1,1))
        f_ip1_j = f_[...,2:,1:-1]
        f_im1_j = f_[...,:-2,1:-1]
        f_i_jp1 = f_[...,1:-1,2:]
        f_i_jm1 = f_[...,1:-1,:-2]
        f = omega * factor * (
                  dxm2 * (cu_ip1_j * f_ip1_j + cu_i_j * f_im1_j)
                + dym2 * (cv_i_jp1 * f_i_jp1 + cv_i_j * f_i_jm1)
                - rhs) \
          + (1 - omega) * f
    return f


def residual(f, rhs, dx, dy, mask, coef_ugrid, coef_vgrid, lambd):
    """ Compute 2D Helmholtz equation with Neumann BC residual:
            res = rhs - Hf
        where Hf is a generalized Helmholtz operator
            Hf = ∇.(c∇f) - λf
        on masked domain.
        The residual is zero iif u is a solution.
    """
    f_ = F.pad(f, (1,1,1,1))
    dx_f = torch.diff(f_[...,1:-1], dim=-2)
    dy_f = torch.diff(f_[...,1:-1,:], dim=-1)
    Hf = mask * (
            torch.diff(coef_ugrid*dx_f, dim=-2) / dx**2 \
          + torch.diff(coef_vgrid*dy_f, dim=-1) / dy**2 \
          - lambd * f)

    return rhs - Hf


def prolong(v, divisor=16):
    """Cell-centered prolongation of the 2D field v."""
    nx, ny = v.shape[-2:]
    v_f = torch.zeros(v.shape[:-2] + (2*nx,2*ny),
                      dtype=v.dtype, device=v.device)
    v_ = F.pad(v, (1,1,1,1))

    # slices
    _2i_2j = (..., slice(0, -1, 2), slice(0, -1, 2))
    _2ip1_2j = (..., slice(1, None, 2), slice(0, -1, 2))
    _2i_2jp1 = (..., slice(0, -1, 2), slice(1, None, 2))
    _2ip1_2jp1 = (..., slice(1, None, 2), slice(1, None, 2))
    _i_j = (..., slice(1, -1), slice(1, -1))
    _ip1_j = (..., slice(2, None), slice(1, -1))
    _im1_j = (..., slice(0, -2), slice(1, -1))
    _i_jp1 = (..., slice(1, -1), slice(2, None))
    _i_jm1 = (..., slice(1, -1), slice(0, -2))
    _im1_jm1 = (..., slice(0, -2), slice(0, -2))
    _ip1_jp1 = (..., slice(2, None), slice(2, None))
    _ip1_jm1 = (..., slice(2, None), slice(0, -2))
    _im1_jp1 = (..., slice(0, -2), slice(2, None))

    # prolongation
    v_f[_2i_2j]     = 9*v_[_i_j] + 3*(v_[_i_jm1] + v_[_im1_j]) + v_[_im1_jm1]
    v_f[_2ip1_2j]   = 9*v_[_i_j] + 3*(v_[_i_jm1] + v_[_ip1_j]) + v_[_ip1_jm1]
    v_f[_2i_2jp1]   = 9*v_[_i_j] + 3*(v_[_i_jp1] + v_[_im1_j]) + v_[_im1_jp1]
    v_f[_2ip1_2jp1] = 9*v_[_i_j] + 3*(v_[_ip1_j] + v_[_i_jp1]) + v_[_ip1_jp1]

    return v_f / divisor


def restrict(v, divisor=4):
    """Cell-centered coarse-grid restriction of the 2D field v."""
    return  (  v[..., :-1:2, :-1:2]
             + v[...,  1::2, :-1:2]
             + v[...,   ::2,  1::2]
             + v[...,  1::2,  1::2]) / divisor




class MG_Helmholtz():
    """
        Multrigrid solver for generalized Helmoltz equations
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
                 dtype=np.float64,
                 device='cpu',
                 mask=None,
                 niter_bottom=-1,
                 use_compilation=True
                 ):

        if lambd is None:
            self.lambd = np.zeros(1, dtype=dtype, device=device)
        else:
            self.lambd = lambd

        if n_levels is None:
            n_levels = min(compute_nlevels(nx), compute_nlevels(ny))
        assert n_levels >= 1, f'at least 1 level needed, got {n_levels}'
        N = 2**(n_levels - 1)
        assert nx % N == 0 and ny % N == 0, \
               f'invalid {n_levels=}, {nx=} and {ny=} must be divisible ' \
               f'by 2**(n_levels-1)={N}'

        print( 'PyTorch multigrid solver '
              f'∇.(c∇f) - λf = rhs, '
              f'λ={self.lambd.view(-1).cpu().numpy()}, {device}, {dtype}, '
              f'n_levels={n_levels}')

        self.dtype = dtype
        self.device = device

        # Grid parameters
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.shape = (self.lambd.shape[0], self.nx, self.ny)

        # pre-/post-smoothing relaxation parameters
        self.omega_pre = torch.tensor(0.95, dtype=self.dtype, device=self.device)
        self.omega_post = torch.tensor(0.95, dtype=self.dtype, device=self.device)

        # Multigrid algo parameters
        self.n_levels = n_levels
        self.max_ite = max_ite
        self.tol = tol
        self.niter_bottom = niter_bottom

        # fine-to-coarse (restrict) and coarse-to-fine (prolong) functions
        self.restrict = restrict
        self.prolong = prolong

        # Mask
        self.compute_mask_hierarchy(mask)

        # Restriction and prolongation divisors
        self.compute_divisor_hierarchy()

        # Divergence coefficient on u- and v-grid
        self.compute_coefficient_hierarchy(coef_ugrid, coef_vgrid)


        # Bottom solving with matrix inversion
        if self.niter_bottom <= 0:
            dx_bottom, dy_bottom = dx*2**(n_levels-1), dy*2**(n_levels-1)
            mask_bottom = self.masks[-1][0]
            coef_ugrid_bottom = self.coefs_ugrid[-1][0]
            coef_vgrid_bottom = self.coefs_vgrid[-1][0]
            i_s, j_s, h_mat = compute_helmholtz_matrix(
                    dx_bottom, dy_bottom, lambd,
                    mask_bottom, coef_ugrid_bottom, coef_vgrid_bottom,
                    dtype, device)
            self.i_s = i_s
            self.j_s = j_s
            self.bottom_inv_mat = torch.linalg.pinv(h_mat.cpu()).to(self.device)


        # precompile torch functions
        if use_compilation:
            f = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
            rhs = torch.zeros_like(f)
            mask = self.masks[0]
            coef_ugrid = self.coefs_ugrid[0]
            coef_vgrid = self.coefs_vgrid[0]
            smooth_args = (f, rhs, self.dx, self.dy, mask,
                        coef_ugrid, coef_vgrid, self.omega_pre, self.lambd)
            self.smooth = torch.jit.trace(jacobi_smoothing, smooth_args)
            residual_args = (f, rhs, self.dx, self.dy, mask,
                            coef_ugrid, coef_vgrid, self.lambd)
            self.residual = torch.jit.trace(residual, residual_args)
            divisor_restrict = torch.ones(nx//2, ny//2, dtype=self.dtype,
                                        device=self.device)
            self.restrict = torch.jit.trace(restrict, (f, divisor_restrict))
            divisor_prolong = torch.ones(2*nx, 2*ny, dtype=self.dtype,
                                        device=self.device)
            self.prolong = torch.jit.trace(prolong, (f, divisor_prolong))
        else:
            self.smooth = jacobi_smoothing
            self.residual = residual
            self.restrict = restrict
            self.prolong = prolong

    def compute_mask_hierarchy(self, mask):
        if mask is None:
            mask = torch.ones(
                    (self.nx, self.ny),
                    dtype=self.dtype, device=self.device)
        assert mask.shape[-2] == self.nx and mask.shape[-1] == self.ny, \
                f'Invalid mask shape {mask.shape}!=({self.nx},{self.ny})'
        mask = mask.unsqueeze(0)
        mask_ugrid, mask_vgrid = compute_mask_uvgrids(mask)
        self.masks = [mask]
        self.masks_ugrid = [mask_ugrid]
        self.masks_vgrid = [mask_vgrid]
        for i in range(1, self.n_levels):
            mask = (self.restrict(self.masks[-1]) > 0.5).type(self.dtype)
            mask_ugrid, mask_vgrid = compute_mask_uvgrids(mask)
            self.masks.append(mask)
            self.masks_ugrid.append(mask_ugrid)
            self.masks_vgrid.append(mask_vgrid)


    def compute_divisor_hierarchy(self):
        self.divisors_restrict = [None]
        for n in range(1, self.n_levels):
            div_restrict = restrict(self.masks[n-1], divisor=1)
            self.divisors_restrict.append(
                torch.max(div_restrict, torch.ones_like(div_restrict)))
        self.divisors_prolong = []
        for n in range(self.n_levels-1):
            div_prolong = prolong(self.masks[n+1], divisor=1)
            self.divisors_prolong.append(
                torch.max(div_prolong, torch.ones_like(div_prolong)))
        self.divisors_prolong.append(None)


    def compute_coefficient_hierarchy(self, coef_ugrid, coef_vgrid):
        nx, ny = self.nx, self.ny
        assert coef_ugrid.shape[-2] == nx+1 and coef_ugrid.shape[-1] == ny, \
               f'Invalid coef shape {coef_ugrid.shape[-2:]}!=({nx+1}, {ny})'
        assert coef_vgrid.shape[-2] == nx and coef_vgrid.shape[-1] == ny+1, \
               f'Invalid coef shape {coef_vgrid.shape[-2:]}!=({nx}, {ny+1})'
        coef = 0.25 * (
                coef_ugrid[...,1:,:] + coef_ugrid[...,:-1,:]
                + coef_vgrid[...,1:]   + coef_vgrid[...,:-1]).unsqueeze(0)
        self.coefs_ugrid = [coef_ugrid.unsqueeze(0) * self.masks_ugrid[0]]
        self.coefs_vgrid = [coef_vgrid.unsqueeze(0) * self.masks_vgrid[0]]
        for i in range(1, self.n_levels):
            coef = restrict(coef, self.divisors_restrict[i]) * self.masks[i]
            self.coefs_ugrid.append(
                F.avg_pool2d(coef, (2,1), stride=(1,1), padding=(1,0))
                * self.masks_ugrid[i])
            self.coefs_vgrid.append(
                F.avg_pool2d(coef, (1,2), stride=(1,1), padding=(0,1))
                * self.masks_vgrid[i])


    def solve(self, rhs, coef_ugrid=None, coef_vgrid=None):
        if coef_ugrid is not None and coef_vgrid is not None:
            self.compute_coefficient_hierarchy(coef_ugrid, coef_vgrid)
        return self.FMG_Helmholtz(rhs, self.dx, self.dy)


    def solve_smooth(self, rhs, coef_ugrid=None, coef_vgrid=None):
        if coef_ugrid is None or coef_vgrid is None:
            coef_ugrid = self.coefs_ugrid[0]
            coef_vgrid = self.coefs_vgrid[0]

        f = torch.zeros_like(rhs)
        for _ in range(1000):
            f = self.smooth(f, rhs, self.dx, self.dy, self.masks[0],
                            coef_ugrid, coef_vgrid, self.omega_pre,
                            self.lambd)
        return f


    def solve_V(self, rhs):
        dx, dy = self.dx, self.dy
        f = torch.zeros_like(rhs)

        nite = 0
        res = self.residual(f, rhs, dx, dy, self.masks[0],
                            self.coefs_ugrid[0], self.coefs_vgrid[0],
                            self.lambd)
        # print(f'init resnorm: {(res.norm()/f.norm()).cpu().item():.3e}')
        # Loop V-cycles until convergence
        while nite < self.max_ite and res.norm()/f.norm() > self.tol:
            f = self.V_cycle(f, rhs, self.n_levels, dx, dy)
            res = self.residual(f, rhs, dx, dy, self.masks[0],
                                self.coefs_ugrid[0],
                                self.coefs_vgrid[0],
                                self.lambd)
            nite += 1
        # print(f'Number or V-cycle: {nite}, residual norm:
        #       f'{(res.norm()/f.norm()).cpu().item():.3E}')

        return f


    def V_cycle(self, f, rhs, n_levels, dx, dy, level=0):
        mask = self.masks[level]
        coef_ugrid = self.coefs_ugrid[level]
        coef_vgrid = self.coefs_vgrid[level]

        # bottom solve
        if level == n_levels-1:
            if self.niter_bottom <= 0: # with matrix inversion
                f = torch.zeros_like(rhs)
                f[..., self.i_s, self.j_s] = torch.einsum(
                        '...l,...lm->...m',
                        rhs[..., self.i_s, self.j_s],
                        self.bottom_inv_mat)
            else: # with Jacobi smoothing
                for _ in range(self.niter_bottom):
                    f = self.smooth(f, rhs, dx, dy, mask, coef_ugrid,
                                    coef_vgrid, self.omega_pre, self.lambd)
            return f

        # Step 1: Relax Au=f on this grid
        f = self.smooth(f, rhs, dx, dy, mask, coef_ugrid, coef_vgrid,
                        self.omega_pre, self.lambd)
        res = self.residual(f, rhs, dx, dy, mask, coef_ugrid, coef_vgrid, self.lambd)

        # Step 2: Restrict residual to coarse grid
        res_coarse = self.restrict(res, self.divisors_restrict[level+1]) \
                     * self.masks[level+1]

        # Step 3: Solve residual equation on the coarse grid. (Recursively)
        eps_coarse = torch.zeros_like(res_coarse)
        eps_coarse = self.V_cycle(eps_coarse, res_coarse,
                                  n_levels, dx*2, dy*2, level=level+1)

        # Step 4: Prolongate eps_coarse to current grid and add to f
        eps = self.prolong(eps_coarse, self.divisors_prolong[level]) * mask
        f += eps

        # Step 5: Relax Au=f on this grid
        f = self.smooth(f, rhs, dx, dy, mask, coef_ugrid, coef_vgrid,
                        self.omega_post, self.lambd)

        return f


    def FMG_Helmholtz(self, rhs, dx, dy):
        """Full Multigrid cycle"""
        rhs_list = [rhs]
        for i in range(1, self.n_levels):
            rhs_list.append(
                self.restrict(rhs_list[-1], self.divisors_restrict[i])
                * self.masks[i])

        # bottom solve
        f = torch.zeros_like(rhs_list[-1])
        rhs_bottom = rhs_list[-1]
        if self.niter_bottom <= 0: # with matrix inversion
            f[..., self.i_s, self.j_s] = torch.einsum(
                    '...l,...lm->...m',
                    rhs_bottom[..., self.i_s, self.j_s],
                    self.bottom_inv_mat)
        else: # smoothing
            k =  2**(self.n_levels - 1)
            coef_ugrid_bottom = self.coefs_ugrid[-1]
            coef_vgrid_bottom = self.coefs_vgrid[-1]
            for _ in range(self.niter_bottom):
                f = self.smooth(f, rhs_bottom, k*dx, k*dy,
                                self.masks[-1], coef_ugrid_bottom,
                                coef_vgrid_bottom,
                                self.omega_pre, self.lambd)

        # Upsample + V-cycle to high-res
        for i in range(2, self.n_levels+1):
            k =  2**(self.n_levels - i)
            f = self.prolong(f, self.divisors_prolong[-i]) * self.masks[-i]
            f = self.V_cycle(f, rhs_list[-i], self.n_levels, k*dx, k*dy, level=self.n_levels-i)

        # Loop V-cycles until convergence
        nite = 0
        res = self.residual(f, rhs, dx, dy, self.masks[0],
                            self.coefs_ugrid[0], self.coefs_vgrid[0], self.lambd)
        while nite < self.max_ite and res.norm()/f.norm() > self.tol:
            f = self.V_cycle(f, rhs, self.n_levels, dx, dy)
            res = self.residual(f, rhs, dx, dy, self.masks[0],
                                self.coefs_ugrid[0], self.coefs_vgrid[0],
                                self.lambd)
            nite += 1
        # print(f'Number or V-cycle: {nite}, {(res.norm()/f.norm()).cpu().item():.3E}')

        return f



if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.size': 16})
    plt.ion()

    def helmholtz(f, dx, dy, coef_ugrid, coef_vgrid, lambd):
        f_ = F.pad(f, (1,1,1,1))
        dx_f = torch.diff(f_[...,1:-1], dim=-2) / dx
        dy_f = torch.diff(f_[...,1:-1,:], dim=-1) / dy
        return torch.diff(coef_ugrid*dx_f, dim=-2) / dx \
             + torch.diff(coef_vgrid*dy_f, dim=-1) / dy - lambd * f


    # device/dtype
    device = 'cpu'
    dtype = torch.float64

    # # Seed
    torch.manual_seed(0)

    # grid
    nx, ny = 64, 64
    dx = torch.tensor(0.1, device=device, dtype=dtype)
    dy = torch.tensor(0.1, device=device, dtype=dtype)

    # Multigrid levels
    n_levels = None

    # masks
    sq_mask = torch.ones(nx, ny, device=device, dtype=dtype)
    circ_mask = torch.ones(nx, ny, device=device, dtype=dtype)
    for i in range(nx):
        for j in range(ny):
            if (i+0.5 - (nx+0.5)/2)**2 + (j+0.5-(ny+0.5)/2)**2 > (nx/2)**2:
                circ_mask[i,j] = 0

    # coefficients
    lambd = torch.ones((1,1,1), device=device, dtype=dtype)
    coef_var_lr =  torch.zeros((1,1,nx//2, ny//2), device=device, dtype=dtype).uniform_(0.9, 1.1)
    coef_var = F.pad(
            F.interpolate(coef_var_lr, (nx, ny), align_corners=False, mode='bilinear'), (1,1,1,1))[0,0]
    coef_cst = torch.ones_like(coef_var)

    coef_cst_ugrid = 0.5*(coef_cst[...,1:,:] + coef_cst[...,:-1,:])[...,1:-1]
    coef_cst_vgrid = 0.5*(coef_cst[...,1:] + coef_cst[...,:-1])[...,1:-1,:]
    coef_var_ugrid = 0.5*(coef_var[...,1:,:] + coef_var[...,:-1,:])[...,1:-1]
    coef_var_vgrid = 0.5*(coef_var[...,1:] + coef_var[...,:-1])[...,1:-1,:]

    for mask, coef_ugrid_, coef_vgrid_, title in [
            (sq_mask, coef_cst_ugrid, coef_cst_vgrid, 'square domain, cst coeff'),
            (sq_mask, coef_var_ugrid, coef_var_vgrid, 'square domain, var coeff'),
            (circ_mask, coef_cst_ugrid, coef_cst_vgrid, 'masked circular domain, cst coeff'),
            (circ_mask, coef_var_ugrid, coef_var_vgrid, 'masked circular domain, var coeff'),
        ]:
        mask_ugrid, mask_vgrid = compute_mask_uvgrids(mask)
        coef_ugrid = coef_ugrid_ * mask_ugrid
        coef_vgrid = coef_vgrid_ * mask_vgrid
        f = torch.DoubleTensor(1, nx, ny).normal_().to(device) * mask
        helm_f = helmholtz(f, dx, dy, coef_ugrid, coef_vgrid, lambd)

        mg = MG_Helmholtz(dx, dy, nx, ny,
                coef_ugrid, coef_vgrid,
                n_levels=n_levels, lambd=lambd,
                device=device, dtype=dtype,
                mask=mask, niter_bottom=8)

        f_rec = mg.solve(helm_f)
        abs_diff = torch.abs(f_rec - f)

        n_plots = 4
        fig, ax = plt.subplots(1, n_plots, figsize=(16,4))
        fig.suptitle(f'Solving ∇.(c∇u) - λu = rhs, {title}')

        fig.colorbar(ax[0].imshow(coef_ugrid.cpu().T, origin='lower'), ax=ax[0])
        ax[0].set_title('coefficient u grid')
        fig.colorbar(ax[1].imshow(coef_vgrid.cpu().T, origin='lower'), ax=ax[1])
        ax[1].set_title('coefficient v grid')
        fig.colorbar(ax[2].imshow(f.cpu()[0].T, origin='lower'), ax=ax[2])
        ax[2].set_title('true f')
        fig.colorbar(ax[3].imshow(abs_diff.cpu()[0].T, origin='lower'), ax=ax[3])
        ax[3].set_title('abs diff')

        [(ax[_].set_xticks([]), ax[_].set_yticks([])) for _ in range(n_plots)]
        fig.tight_layout()
