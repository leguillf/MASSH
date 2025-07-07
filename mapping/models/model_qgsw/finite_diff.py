"""Finite difference operators in pytorch,
Louis Thiry, 6 march 2023."""

import jax.numpy as jnp 

def interp_TP(f):
    return 0.25 *(f[...,1:,1:] + f[...,1:,:-1] + f[...,:-1,1:] + f[...,:-1,:-1])

def interp_TP_inv(g):
    """
    Approximate inversion of:
        interp_TP(f) = 0.25*(f[...,1:,1:] + f[...,1:,:-1] + f[...,:-1,1:] + f[...,:-1,:-1])
    by expanding g to the shape of f with repeated values.
    """
    # Shape: (..., x, y) -> (..., x+1, y+1)
    f_shape = g.shape[:-2] + (g.shape[-2]+1, g.shape[-1]+1)
    f_est = jnp.zeros(f_shape)

    # Distribute g back:
    f_est = f_est.at[..., 1:, 1:].add(g)
    f_est = f_est.at[..., 1:, :-1].add(g)
    f_est = f_est.at[..., :-1, 1:].add(g)
    f_est = f_est.at[..., :-1, :-1].add(g)

    # Since each cell in f contributes to 4 cells in g,
    # and we summed back, we rescale by 0.25 to approximate inversion
    return 0.25 * f_est

def comp_ke(u, U, v, V):
    u_sq = u * U
    v_sq = v * V
    return 0.25*(u_sq[...,1:,:] + u_sq[...,:-1,:] + v_sq[...,1:] + v_sq[...,:-1])

def laplacian(f, dx, dy):
    return (f[...,2:,1:-1] - 2*f[...,1:-1,1:-1] + f[...,:-2,1:-1]) / dx**2 \
         + (f[...,1:-1,2:] - 2*f[...,1:-1,1:-1] + f[...,1:-1,:-2]) / dy**2


def grad_perp(f):
    """Orthogonal gradient"""
    return f[...,:-1] - f[...,1:], f[...,1:,:] - f[...,:-1,:]

def div_nofluxbc(flux_x, flux_y):
    # flux_y padding on width (last dimension)
    flux_y_padded = jnp.pad(flux_y, ((0,0), (0,0), (0,0), (1,1)), mode='edge')
    div_y = jnp.diff(flux_y_padded, axis=-1)

    # flux_x padding on height (second to last dimension)
    flux_x_padded = jnp.pad(flux_x, ((0,0), (0,0), (1,1), (0,0)), mode='edge')
    div_x = jnp.diff(flux_x_padded, axis=-2)

    return div_y + div_x
