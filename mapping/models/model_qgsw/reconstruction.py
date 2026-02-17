"""
Linear and WENO reconstructions.
Louis Thiry, 2023
"""
import jax.numpy as jnp 

# Epsilon for WENO smoothness indicators.
# Standard value 1e-14 prevents division by zero in the forward model,
# but causes adjoint blow-up since d/d(beta)[tau/(beta+eps)] ~ 1/(beta+eps)^2.
# A larger value stabilizes the adjoint with negligible impact on forward accuracy.
# Use 1e-4 for float32 (safe: 1/eps^2 = 1e8, within f32 range).
# Use 1e-6 for float64.
WENO_EPS = 1e-4

def smooth_abs(x):
    """Smooth approximation of |x|, differentiable everywhere including x=0."""
    return jnp.sqrt(x**2 + WENO_EPS**2)


def linear2_centered(qm, qp):
    """
    2-points linear centered reconstruction:

    qm--x--qp
    ^      ^
    """
    return 0.5 * (qm + qp)


def linear2_left(qm, qp):
    """
    2-points linear left-biased reconstruction:

    qm--x--qp
    ^      X
    """
    return qm


def linear4_centered(qmm, qm, qp, qpp):
    """
    4-points linear centered reconstruction:

    qmm-----qm--x--qp-----qpp
    ^       ^      ^      ^
    """
    return -1./12.*qmm + 7./12.*qm + 7./12.*qp - 1./12.*qpp


def linear4_left(qmm, qm, qp, qpp):
    """
    4-points linear left-biased reconstruction:

    qmm-----qm--x--qp-----qpp
    ^       ^      ^      X
    """
    return -1./6.*qmm + 5./6.*qm + 1./3.*qp


def linear6_left(qmmm, qmm, qm, qp, qpp, qppp):
    """
    6-points linear left-biased stencil reconstruction

    qmmm----qmm-----qm--x--qp----qpp----qppp
    ^       ^       ^      ^     ^      X
    """
    return 1./30.*qmmm - 13./60.*qmm + 47./60.*qm + 9./20.*qp - 1/20 *qpp


def wenojs4_left(qmm, qm, qp, qpp):
    """
    4-points weno-JS left-biased reconstruction,
        after https://doi.org/10.1006/jcph.1996.0130
    qmm-----qm--x--qp-----qpp
    ^       ^      ^      X
    """
    qi1 = -1./2.*qmm + 3./2.*qm
    qi2 = 1./2.*(qm + qp)

    beta1 = (qm-qmm)**2
    beta2 = (qp-qm)**2

    g1, g2 = 1./3., 2./3.
    w1 = g1 / (beta1+WENO_EPS)**2
    w2 = g2 / (beta2+WENO_EPS)**2

    qi_weno = (w1*qi1 + w2*qi2) / (w1 + w2)

    return qi_weno


def wenoz4_left(qmm, qm, qp, qpp):
    """
    4-points weno-Z left-biased reconstruction,
        after https://doi.org/10.1016/j.jcp.2007.11.038
    qmm-----qm--x--qp-----qpp
    ^       ^      ^      X
    """
    qi1 = -1./2.*qmm + 3./2.*qm
    qi2 = 1./2.*(qm + qp)

    beta1 = (qm-qmm)**2
    beta2 = (qp-qm)**2
    tau = smooth_abs(beta2-beta1)

    g1, g2 = 1./3., 2./3.
    w1 = g1 * (1. + tau / (beta1 + WENO_EPS))
    w2 = g2 * (1. + tau / (beta2 + WENO_EPS))

    qi_weno = (w1*qi1 + w2*qi2) / (w1 + w2)

    return qi_weno


def wenojs6_left(qmmm, qmm, qm, qp, qpp, qppp):
    """
    6-points weno-JS left-biased reconstruction,
        after https://doi.org/10.1006/jcph.1996.0130

    qmmm----qmm-----qm--x--qp----qpp----qppp
    ^       ^       ^      ^     ^      X
    """
    qi1 = 1./3.*qmmm - 7./6.*qmm + 11./6.*qm
    qi2 = -1./6.*qmm + 5./6.*qm + 1./3.*qp
    qi3 = 1./3.*qm + 5./6.*qp - 1./6.*qpp

    k1, k2 = 13./12., 0.25
    beta1 = k1 * (qmmm-2*qmm+qm)**2 + k2 * (qmmm-4*qmm+3*qm)**2
    beta2 = k1 * (qmm-2*qm+qp)**2  + k2 * (qmm-qp)**2
    beta3 = k1 * (qm-2*qp+qpp)**2 + k2 * (3*qm-4*qp+qpp)**2

    g1, g2, g3 = 0.1, 0.6, 0.3
    w1 = g1 / (beta1+WENO_EPS)**2
    w2 = g2 / (beta2+WENO_EPS)**2
    w3 = g3 / (beta3+WENO_EPS)**2

    qi_weno = (w1*qi1 + w2*qi2 + w3*qi3) / (w1 + w2 + w3)

    return qi_weno


def wenoz6_left(qmmm, qmm, qm, qp, qpp, qppp):
    """
    6-points weno-Z left-biased reconstruction,
        after https://doi.org/10.1016/j.jcp.2007.11.038

    qmmm----qmm-----qm--x--qp----qpp----qppp
    ^       ^       ^      ^     ^      X
    """
    qi1 = 1./3.*qmmm - 7./6.*qmm + 11./6.*qm
    qi2 = -1./6.*qmm + 5./6.*qm + 1./3.*qp
    qi3 = 1./3.*qm + 5./6.*qp - 1./6.*qpp

    k1, k2 = 13./12., 0.25
    beta1 = k1 * (qmmm-2*qmm+qm)**2 + k2 * (qmmm-4*qmm+3*qm)**2
    beta2 = k1 * (qmm-2*qm+qp)**2  + k2 * (qmm-qp)**2
    beta3 = k1 * (qm-2*qp+qpp)**2 + k2 * (3*qm-4*qp+qpp)**2

    tau5 = smooth_abs(beta1 - beta3)

    g1, g2, g3 = 0.1, 0.6, 0.3
    w1 = g1 * (1 + tau5 / (beta1 + WENO_EPS))
    w2 = g2 * (1 + tau5 / (beta2 + WENO_EPS))
    w3 = g3 * (1 + tau5 / (beta3 + WENO_EPS))


    qi_weno = (w1*qi1 + w2*qi2 + w3*qi3) / (w1 + w2 + w3)

    return qi_weno
