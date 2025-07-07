"""
Velocity-sign biased flux computations.
Louis Thiry, 2023
"""
import jax.numpy as jnp 


def stencil_2pts(q, dim):
    n = q.shape[dim % q.ndim]
    dim = dim % q.ndim
    slc0 = [slice(None)] * q.ndim
    slc1 = [slice(None)] * q.ndim
    slc0[dim] = slice(0, n - 1)
    slc1[dim] = slice(1, n)
    return q[tuple(slc0)], q[tuple(slc1)]

def stencil_4pts(q, dim):
    n = q.shape[dim % q.ndim]
    dim = dim % q.ndim
    return tuple(
        q[tuple(slc)]
        for slc in (
            [slice(None)] * dim + [slice(i, n - 3 + i)] + [slice(None)] * (q.ndim - dim - 1)
            for i in range(4)
        )
    )

def stencil_6pts(q, dim):
    n = q.shape[dim % q.ndim]
    dim = dim % q.ndim
    return tuple(
        q[tuple(slc)]
        for slc in (
            [slice(None)] * dim + [slice(i, n - 5 + i)] + [slice(None)] * (q.ndim - dim - 1)
            for i in range(6)
        )
    )

def flux(q, u,
        dim,
        n_points,
        rec_func_2,
        rec_func_4,
        rec_func_6,
        mask_2,
        mask_4,
        mask_6,
        ):
    # positive and negative velocities
    u_pos = jnp.clip(u, a_min=0)
    u_neg = u - u_pos

    # 2-points reconstruction
    q_stencil2 = stencil_2pts(q, dim)
    qi2_pos = rec_func_2(*q_stencil2)
    qi2_neg = rec_func_2(*q_stencil2[::-1])

    if n_points == 2:
        return u_pos*qi2_pos + u_neg*qi2_neg

    # 4-points reconstruction
    if dim == -1:
        pad = ((0,0), (0,0), (0,0), (1,1))
    else:
        pad = ((0,0), (0,0), (1,1), (0,0))

    q_stencil4 = stencil_4pts(q, dim)
    qi4_pos = jnp.pad(rec_func_4(*q_stencil4), pad)
    qi4_neg = jnp.pad(rec_func_4(*q_stencil4[::-1]), pad)

    if n_points == 4:
        return u_pos * (mask_2 * qi2_pos + mask_4*qi4_pos) \
             + u_neg * (mask_2 * qi2_neg + mask_4*qi4_neg)

    # 6-points reconstruction
    if dim == -1:
        pad = ((0,0), (0,0), (0,0), (2,2))
    else:
        pad = ((0,0), (0,0), (2,2), (0,0))

    q_stencil6 = stencil_6pts(q, dim)
    qi6_pos = jnp.pad(rec_func_6(*q_stencil6), pad)
    qi6_neg = jnp.pad(rec_func_6(*q_stencil6[::-1]), pad)

    if n_points == 6:
        return u_pos * (mask_2 * qi2_pos + mask_4*qi4_pos + mask_6*qi6_pos) \
             + u_neg * (mask_2 * qi2_neg + mask_4*qi4_neg + mask_6*qi6_neg)

    # raise NotImplementedError(f'flux computations implemented for '
                              # f'2, 4, 6 points stencils, got {n_points}')
