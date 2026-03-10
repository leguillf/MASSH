"""
Spectral 2D Helmholtz equation solver on rectangular and non-rectangular domain.
  - Colocated Dirichlet BC with DST-I  (type-I discrete sine transform)
  - Staggered Neumann   BC with DCT-II (type-II discrete consine transform)
  - Non-rectangular domains emmbedded in rectangular domains with a mask.
  - Capacitance matrix method for non-rectangular domains
Louis Thiry, 2023.
"""
import sys 
sys.path.insert(0, '../../src') # add src to path to import modules
from src.config import USE_FLOAT64
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp 
import matplotlib.pyplot as plt
from tools import avg_pool2d

jax.config.update("jax_enable_x64", USE_FLOAT64)


def compute_laplace_dctII(nx, ny, dx, dy, arr_kwargs):
    """DCT-II of standard 5-points laplacian on uniform grid"""
    x, y = jnp.meshgrid(jnp.arange(nx, **arr_kwargs),
                          jnp.arange(ny, **arr_kwargs),
                          indexing='ij')
    return 2*(jnp.cos(jnp.pi/nx*x) - 1)/dx**2 \
         + 2*(jnp.cos(jnp.pi/ny*y) - 1)/dy**2


def dctII(x, exp_vec):
    """
    1D forward type-II discrete cosine transform (DCT-II)
    using fft and precomputed auxillary vector exp_vec.
    """
    v = jnp.concatenate([x[...,::2], jnp.flip(x, axis=(-1,))[...,::2]], axis=-1)
    V = jnp.fft.fft(v)
    return (V*exp_vec).real


def idctII(x, iexp_vec):
    """
    1D inverse type-II discrete cosine transform (DCT-II)
    using fft and precomputed auxillary vector iexp_vec.
    """
    N = x.shape[-1]
    x_rev = jnp.flip(x, axis=(-1,))[...,:-1]
    v = jnp.concatenate([x[...,0:1],
            iexp_vec[...,1:N]*(x[...,1:N]-1j*x_rev)], axis=-1) / 2
    V = jnp.fft.ifft(v)
    y = jnp.zeros_like(x)
    y = y.at[...,::2].set(V[...,:N//2].real)
    y = y.at[...,1::2].set(jnp.flip(V, axis=(-1,))[...,:N//2].real)
    return y


def dctII2D(x, exp_vec_x, exp_vec_y):
    """2D forward DCT-II. Works for any number of leading batch dimensions."""
    return dctII(
            dctII(x, exp_vec_y).swapaxes(-1, -2),
            exp_vec_x).swapaxes(-1, -2)


def idctII2D(x, iexp_vec_x, iexp_vec_y):
    """2D inverse DCT-II. Works for any number of leading batch dimensions."""
    return idctII(
            idctII(x, iexp_vec_y).swapaxes(-1, -2),
            iexp_vec_x).swapaxes(-1, -2)


def compute_dctII_exp_vecs(N, dtype):
    """Compute auxillary exp_vec and iexp_vec used in
    fast DCT-II computations with FFTs."""
    N_range = jnp.arange(N, dtype=dtype)
    exp_vec = 2 * jnp.exp(-1j*jnp.pi*N_range/(2*N))
    iexp_vec = jnp.exp(1j*jnp.pi*N_range/(2*N))
    return exp_vec, iexp_vec


def solve_helmholtz_dctII(rhs, helmholtz_dctII,
        exp_vec_x, exp_vec_y,
        iexp_vec_x, iexp_vec_y):
    """Solves Helmholtz equation with DCT-II fast diagonalisation."""
    rhs_dctII = dctII2D(rhs.astype(helmholtz_dctII.dtype), exp_vec_x, exp_vec_y)
    return idctII2D(rhs_dctII / helmholtz_dctII, iexp_vec_x, iexp_vec_y
            ).astype(rhs.dtype)


def _dstI1D(x, norm='ortho'):
    """1D type-I discrete sine transform (DST-I), forward and inverse
    since DST-I is auto-inverse."""

    # 1) pad only the last axis by 1 on each side
    pad_width = [(0, 0)] * (x.ndim - 1) + [(1, 1)]
    x_padded = jnp.pad(x, pad_width, mode='constant', constant_values=0)

    # 2) multiply by -1j
    x_padded = -1j * x_padded

    # 3) inverse real FFT along the last axis
    #    JAX’s irfft signature is (x, n=None, axis=-1, norm=None)
    y = jnp.fft.irfft(x_padded, axis=-1, norm=norm)

    # 4) slice off the extra padding to match PyTorch’s [...,1:orig_len+1]
    return y[..., 1 : x.shape[-1] + 1]


def _dstI2D(x, norm='ortho'):
    """2D DST-I."""
    return jnp.swapaxes(jnp.swapaxes(dstI1D(x, norm=norm), -1, -2), -1, -2)

def dstI1D(x, norm='ortho'):
    """1D type-I discrete sine transform (DST-I) in JAX."""
    # Pad (1,1) on the last axis
    x_padded = jnp.pad(x, [(0,0)] * (x.ndim - 1) + [(1,1)])
    # Compute irfft
    result = jnp.fft.irfft(-1j * x_padded, axis=-1, norm=norm)[..., 1:x.shape[-1]+1]
    return result

def dstI2D(x, norm='ortho'):
    """2D DST-I in JAX."""
    x = dstI1D(x, norm=norm)
    x = jnp.swapaxes(x, -1, -2)
    x = dstI1D(x, norm=norm)
    x = jnp.swapaxes(x, -1, -2)
    return x

def compute_laplace_dstI(nx, ny, dx, dy, arr_kwargs):
    """Type-I discrete sine transform of the usual 5-points
    discrete laplacian operator on uniformly spaced grid."""
    x, y = jnp.meshgrid(jnp.arange(1,nx, **arr_kwargs),
                          jnp.arange(1,ny, **arr_kwargs),
                          indexing='ij')
    return 2*(jnp.cos(jnp.pi/nx*x) - 1)/dx**2 \
         + 2*(jnp.cos(jnp.pi/ny*y) - 1)/dy**2


def solve_helmholtz_dstI(rhs, helmholtz_dstI):
    """Solves 2D Helmholtz equation with DST-I fast diagonalization."""
    return jnp.pad(
                dstI2D(dstI2D(rhs.astype(helmholtz_dstI.dtype))/ helmholtz_dstI), 
                ((0,0), (0,0), (1,1), (1,1)), mode='edge'
                ).astype(rhs.dtype)


def compute_capacitance_matrices(helmholtz_dstI, bound_xids, bound_yids):
    nl  = helmholtz_dstI.shape[-3]
    M = bound_xids.shape[0]

    # compute G matrices
    inv_cap_matrices = jnp.zeros((nl, M, M), dtype=jnp.float64)
    rhs = jnp.zeros(helmholtz_dstI.shape[-3:], dtype=jnp.float64,
                      )
    for m in range(M):
        rhs = jnp.zeros_like(rhs)
        rhs = rhs.at[..., bound_xids[m], bound_yids[m]].set(1)
        sol = dstI2D(dstI2D(rhs) / helmholtz_dstI.astype(jnp.float64))
        # Extract values at boundary points and ensure correct shape
        boundary_values = sol[...,bound_xids, bound_yids]
        if boundary_values.ndim > 2:
            boundary_values = boundary_values.squeeze(axis=0)
        inv_cap_matrices = inv_cap_matrices.at[:,m].set(boundary_values)

    # invert G matrices to get capacitance matrices
    cap_matrices = jnp.zeros_like(inv_cap_matrices)
    for l in range(nl):
        cap_matrices = cap_matrices.at[l].set(jnp.linalg.inv(inv_cap_matrices[l]))

    return cap_matrices


def solve_helmholtz_dstI_cmm(rhs, helmholtz_dstI,
                            cap_matrices, bound_xids, bound_yids,
                            mask):
    sol_1 = dstI2D(
                dstI2D(rhs.astype(helmholtz_dstI.dtype)) / helmholtz_dstI
            ).astype(rhs.dtype)
    alphas = jnp.einsum(
        '...ij,...j->...i', cap_matrices, -sol_1[..., bound_xids, bound_yids])

    rhs_2 = jnp.zeros_like(rhs)
    rhs_2 = rhs_2.at[..., bound_xids, bound_yids].set(alphas)

    return solve_helmholtz_dstI(rhs + rhs_2, helmholtz_dstI) * mask


class HelmholtzNeumannSolver:
    def __init__(self, nx, ny, dx, dy, lambd, dtype, mask=None):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.lambd = lambd
        self.dtype = dtype

        # helmholtz dct-II
        self.helmholtz_dctII = compute_laplace_dctII(
                nx, ny, dx, dy, {'dtype':dtype}) \
                - lambd

        # auxillary vectors for DCT-II computations
        exp_vec_x, iexp_vec_x = compute_dctII_exp_vecs(nx, dtype)
        exp_vec_y, iexp_vec_y = compute_dctII_exp_vecs(ny, dtype)
        self.exp_vec_x = jnp.expand_dims(jnp.expand_dims(exp_vec_x, 0), 0)
        self.iexp_vec_x = jnp.expand_dims(jnp.expand_dims(iexp_vec_x, 0), 0)
        self.exp_vec_y = jnp.expand_dims(jnp.expand_dims(exp_vec_y, 0), 0)
        self.iexp_vec_y = jnp.expand_dims(jnp.expand_dims(iexp_vec_y, 0), 0)

        # mask
        if mask is not None:
            shape = mask.shape[0], mask.shape[1]
            assert shape == (nx, ny), f'Invalid mask {shape=} != nx, ny {nx, ny}'
            self.mask = jnp.expand_dims(mask, 0).astype(dtype)
        else:
            self.mask = jnp.ones(1, nx, ny, dtype=self.dtype)
        self.not_mask = 1 - self.mask

        # mask on u- and v-grid
        self.mask_u = (avg_pool2d(self.mask, kernel_size=(2,1), stride=(1,1), padding=(1,0), divisor_override=1) > 1.5).astype(self.dtype)
        self.mask_v = (avg_pool2d(self.mask, kernel_size=(1,2), stride=(1,1), padding=(0,1), divisor_override=1) > 1.5).astype(self.dtype)
        

        # irregular boundary indices
        mask_neighbor_x = self.mask * jnp.pad(
                    avg_pool2d(self.mask, (3,1), stride=(1,1), divisor_override=1) < 2.5
                , pad_width=((0,0), (1,1), (0,0)))
        mask_neighbor_y = self.mask * jnp.pad(
                    avg_pool2d(self.mask, (1,3), stride=(1,1), divisor_override=1) < 2.5
                , pad_width=((0,0), (0,0), (1,1)))
        self.mask_irrbound = jnp.logical_or(
                mask_neighbor_x, mask_neighbor_y)
        self.irrbound_xids, self.irrbound_yids = jnp.where(self.mask_irrbound[0])

        # compute capacitance matrix
        self.compute_capacitance_matrix()


    def helmholtz_reg_domain(self, f):
        pad_width = [(0, 0)] * (f.ndim - 2) + [(1, 1), (1, 1)]
        f_ = jnp.pad(f, pad_width=pad_width, mode='edge')
        dxx_f = (f_[...,2:,1:-1] + f_[...,:-2,1:-1] - 2*f_[...,1:-1,1:-1]) \
                / self.dx**2
        dyy_f = (f_[...,1:-1,2:] + f_[...,1:-1,:-2] - 2*f_[...,1:-1,1:-1]) \
                / self.dy**2
        return dxx_f + dyy_f - self.lambd * f


    def helmholtz(self, f):
        if len(self.irrbound_xids) == 0:
            return self.helmholtz_reg_domain(f)

        pad_width = [(0, 0)] * (f.ndim - 2) + [(1, 1), (1, 1)]
        f_ = jnp.pad(f, pad_width=pad_width, mode='edge')
        dx_f = jnp.diff(f_[...,1:-1], axis=-2) / self.dx
        dy_f = jnp.diff(f_[...,1:-1,:], axis=-1) / self.dy
        dxx_f = jnp.diff(dx_f*self.mask_u, axis=-2) / self.dx
        dyy_f = jnp.diff(dy_f*self.mask_v, axis=-1) / self.dy

        return (dxx_f + dyy_f - self.lambd * f) * self.mask


    def compute_capacitance_matrix(self):
        M = len(self.irrbound_xids)
        if M == 0:
            self.cap_matrix = None
            return

        # compute inverse capacitance matrice
        inv_cap_matrix = jnp.zeros(
                (M, M), dtype=jnp.float64)
        for m in range(M):
            v = jnp.zeros(M, dtype=jnp.float64)
            v = v.at[m].set(1)
            inv_cap_matrix = inv_cap_matrix.at[:,m].set(
                (v - self.V_T(self.G(self.U(v))))[0])

        # invert on cpu
        cap_matrix = jnp.linalg.inv(inv_cap_matrix)

        # convert to dtype and transfer to device
        self.cap_matrix = cap_matrix.astype(self.dtype)


    def U(self, v):
        Uv = jnp.zeros_like(self.mask)
        Uv = Uv.at[...,self.irrbound_xids, self.irrbound_yids].set(v)
        return Uv


    def V_T(self, field):
        return (
            self.helmholtz_reg_domain(field)
          - self.helmholtz(field)
          )[..., self.irrbound_xids, self.irrbound_yids]


    def G(self, field):
        return solve_helmholtz_dctII(field, self.helmholtz_dctII,
                self.exp_vec_x, self.exp_vec_y, self.iexp_vec_x,
                self.iexp_vec_y)


    def solve(self, rhs):
        GF = self.G(rhs)
        if len(self.irrbound_xids) == 0:
            return GF
        V_TGF = self.V_T(GF)
        rho = jnp.einsum(
            'ij,...j->...i', self.cap_matrix, V_TGF)
        return GF + self.G(self.U(rho))



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    matplotlib.rcParams.update({'font.size': 18})
    plt.ion()

    dtype = jnp.float64

    # grid
    N = 4
    nx, ny = 2*2**(N-1), 2*2**(N-1)
    shape = (nx, ny)
    L = 2000e3
    xc = jnp.linspace(-L, L, nx+1, dtype=dtype)
    yc = jnp.linspace(-L, L, ny+1, dtype=dtype)
    xx, yy = jnp.meshgrid(xc, yc, indexing='ij')
    dx = xc[1] - xc[0]
    dy = yc[1] - yc[0]

    # Helmholtz eq.
    lambd = 1e-2 * jnp.ones([1,1,1],dtype=dtype)
    helmholtz_dirichletbc = lambda f, dx, dy, lambd: \
          (f[...,2:,1:-1] + f[...,:-2,1:-1] - 2*f[...,1:-1,1:-1])/dx**2 \
        + (f[...,1:-1,2:] + f[...,1:-1,:-2] - 2*f[...,1:-1,1:-1])/dy**2 \
        - lambd * f[...,1:-1,1:-1]
    helmholtz_dstI = compute_laplace_dstI(
            nx, ny, dx, dy, {'dtype':dtype}) \
            - lambd


    # Rectangular domain
    frect = jnp.zeros([1, nx+1, ny+1], dtype=dtype)
    frect = frect.at[...,1:-1,1:-1].set(
        jax.random.normal(jax.random.PRNGKey(0), shape=frect[...,1:-1,1:-1].shape))
    Hfrect = helmholtz_dirichletbc(frect, dx, dy, lambd)
    frect_r = solve_helmholtz_dstI(Hfrect, helmholtz_dstI)
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    ax[0].set_title('f')
    fig.colorbar(ax[0].imshow(frect[0].cpu().T, origin='lower'), ax=ax[0])
    ax[1].set_title('|f - f_r|')
    fig.colorbar(ax[1].imshow(jnp.abs(frect - frect_r)[0].cpu().T, origin='lower'), ax=ax[1])
    fig.suptitle('Solving Dirichlet-BC Helmholtz equation on square domain with DSTI')
    fig.savefig('helmholtz_rect.png')
    fig.tight_layout()


    # Circular domain
    mask = (1 > ((xx/L)**2 + (yy/L)**2)).type(dtype)
    mask[[0,-1],:] = 0
    mask[[0,-1]] = 0
    domain_neighbor = \
        F.avg_pool2d(mask.reshape((1,1)+mask.shape), kernel_size=3, stride=1, padding=0)[0,0] > 0
    irrbound_xids, irrbound_yids = jnp.where(
            jnp.logical_and(mask[1:-1,1:-1] < 0.5, domain_neighbor))
    cap_matrices = compute_capacitance_matrices(
            helmholtz_dstI, irrbound_xids,
            irrbound_yids)

    fcirc = mask * jnp.zeros_like(mask).normal_().unsqueeze(0)
    Hfcirc = helmholtz_dirichletbc(fcirc, dx, dy, lambd) * mask[1:-1,1:-1]
    fcirc_r = solve_helmholtz_dstI_cmm(Hfcirc, helmholtz_dstI,
                            cap_matrices, irrbound_xids,
                            irrbound_yids, mask)

    palette = plt.cm.bwr.with_extremes(bad='grey')
    fig, ax = plt.subplots(1,2, figsize=(18,9))
    ax[0].set_title('$f$')
    vM = fcirc[0].abs().max().cpu().item()
    # fcirc_ma = np.ma.masked_where((1-mask).cpu().numpy(), fcirc[0].cpu().numpy())
    fcirc_ma = fcirc[0].cpu().numpy()
    fig.colorbar(ax[0].imshow(fcirc_ma.T, vmin=-vM, vmax=vM, origin='lower', cmap=palette), ax=ax[0])
    ax[1].set_title('$f - f_{\\rm inv}$')
    diff =( fcirc - fcirc_r)[0].cpu().numpy()
    vM = np.abs(diff).max()
    # diff_ma = np.ma.masked_where((1-mask).cpu().numpy(), diff)
    diff_ma =diff
    fig.colorbar(ax[1].imshow(diff_ma.T, vmin=-vM, vmax=vM, origin='lower', cmap=palette), ax=ax[1])
    fig.suptitle('Solving Dirichlet-BC Helmholtz eq. $\\Delta f - f = r$ on circular domain with CMM and DST-I')
    ax[0].set_xticks([]), ax[1].set_xticks([]), ax[0].set_yticks([]), ax[1].set_yticks([])
    fig.savefig('helmholtz_circular.png')
    fig.tight_layout()


    ## Neumann BC
    dx, dy = 20000, 20000
    lambd = jnp.ones(1,1,1).type(dtype) / dx * dy

    N = 32
    nx, ny = N//2, N
    f1 = jnp.zeros(1, nx, ny, dtype=dtype).normal_()
    mask1 = jnp.ones(nx, ny, dtype=dtype)

    solver1 = HelmholtzNeumannSolver(
            nx, ny, dx, dy, lambd, dtype, mask=mask1)

    H_f1 = solver1.helmholtz(f1)
    f1_r = solver1.solve(H_f1)

    vM = max(f1.abs().cpu().max().item(), f1_r.cpu().max().item())
    diff = (f1 - f1_r) * mask1
    vM2 = diff.abs().cpu().max().item()

    fig, ax = plt.subplots(1,3, figsize=(16,6))
    ax[0].set_title('f')
    fig.colorbar(ax[0].imshow(f1[0].cpu().T, vmin=-vM, vmax=vM, cmap='bwr', origin='lower'), ax=ax[0])
    ax[1].set_title('f_r')
    fig.colorbar(ax[1].imshow(f1_r[0].cpu().T, vmin=-vM, vmax=vM, cmap='bwr', origin='lower'), ax=ax[1])
    ax[2].set_title('|f - f_r|')
    fig.colorbar(ax[2].imshow(diff[0].cpu().T, vmin=-vM2, vmax=vM2, cmap='bwr', origin='lower'), ax=ax[2])
    fig.suptitle('Neumann-BC, solving Helmholtz eq. with DCT-II')
    ax[0].set_xticks([]), ax[1].set_xticks([]), ax[2].set_xticks([])
    ax[0].set_yticks([]), ax[1].set_yticks([]), ax[2].set_yticks([])
    fig.savefig('Neumann-BC_1.png')
    fig.tight_layout()

    #
    nx, ny = N, N
    f2 = torch.zeros(1, nx, ny, dtype=dtype, device=device)
    f2[:,N//2:,:] = f1
    mask2 = torch.ones(nx, ny, dtype=dtype, device=device)
    mask2[:N//2,:] = 0

    solver2 = HelmholtzNeumannSolver(
            nx, ny, dx, dy, lambd, dtype, device, mask=mask2)

    H_f2 = solver2.helmholtz(f2)
    f2_r = solver2.solve(H_f2)

    vM = max(f2.abs().cpu().max().item(), f2_r.cpu().max().item())
    diff = (f2 - f2_r) * mask2
    vM2 = diff.abs().cpu().max().item()

    fig, ax = plt.subplots(1,3, figsize=(16,6))
    ax[0].set_title('f')
    fig.colorbar(ax[0].imshow(f2[0].cpu().T, vmin=-vM, vmax=vM, cmap='bwr', origin='lower'), ax=ax[0])
    ax[1].set_title('f_r')
    fig.colorbar(ax[1].imshow(f2_r[0].cpu().T, vmin=-vM, vmax=vM, cmap='bwr', origin='lower'), ax=ax[1])
    ax[2].set_title('|f - f_r|')
    fig.colorbar(ax[2].imshow(diff[0].cpu().T, vmin=-vM2, vmax=vM2, cmap='bwr', origin='lower'), ax=ax[2])
    fig.suptitle('Neumann-BC, solving Helmholtz eq. with DCT-II')
    ax[0].set_xticks([]), ax[1].set_xticks([]), ax[2].set_xticks([])
    ax[0].set_yticks([]), ax[1].set_yticks([]), ax[2].set_yticks([])
    fig.savefig('Neumann-BC_2.png')
    fig.tight_layout()

