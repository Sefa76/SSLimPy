import numpy as np
from numba import njit, prange
from SSLimPy.utils.utils import *

#########################
# Fundamental functions #
#########################


@njit("(float64,float64,float64,float64,float64,float64)", fastmath=True)
def alpha(k1, mu1, ph1, k2, mu2, ph2):
    k12, mu12, ph12 = addVectors(k1, mu1, ph1, k2, mu2, ph2)
    return scalarProduct(k12, mu12, ph12, k1, mu1, ph1) / k1**2


@njit("(float64,float64,float64,float64,float64,float64)", fastmath=True)
def beta(k1, mu1, ph1, k2, mu2, ph2):
    k12, mu12, ph12 = addVectors(k1, mu1, ph1, k2, mu2, ph2)
    k1pk2 = scalarProduct(k1, mu1, ph1, k2, mu2, ph2)
    return k12**2 * k1pk2 / (2 * k1**2 * k2**2)


@njit(
    "(float64, float64, float64," + "float64, float64, float64)",
    fastmath=True,
)
def Galileon2(k1, mu1, ph1, k2, mu2, ph2):
    """Second Galileon in Fourier space"""
    muC12 = scalarProduct(k1, mu1, ph1, k2, mu2, ph2) / (k1 * k2)
    return muC12**2 - 1


@njit(
    "(float64, float64, float64,"
    + "float64, float64, float64,"
    + "float64, float64, float64)",
    fastmath=True,
)
def Galileon3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """Third Galileon in Fourier space"""
    muC12 = scalarProduct(k1, mu1, ph1, k2, mu2, ph2) / (k1 * k2)
    muC23 = scalarProduct(k2, mu2, ph2, k3, mu3, ph3) / (k2 * k3)
    muC13 = scalarProduct(k1, mu1, ph1, k3, mu3, ph3) / (k1 * k3)
    return 1 + -(muC12**2) - muC23**2 - muC13**2 + 2 * muC12 * muC23 * muC13


######################################
# (Symmetrised) Mode-Coupling Kernels #
######################################


@njit("(float64,float64,float64,float64,float64,float64)", fastmath=True)
def vF2(k1, mu1, ph1, k2, mu2, ph2):
    """Symmetrised F2 kernel"""
    k, _, _ = addVectors(k1, mu1, ph1, k2, mu2, ph2)
    if np.isclose(k, 0):
        return 0
    else:
        k1pk2 = scalarProduct(k1, mu1, ph1, k2, mu2, ph2)
        F2 = (
            5 / 7
            + 1 / 2 * (1 / k1**2 + 1 / k2**2) * k1pk2
            + 2 / 7 * k1pk2**2 / (k1 * k2) ** 2
        )
        return F2


@njit("(float64,float64,float64,float64,float64,float64)", fastmath=True)
def vG2(k1, mu1, ph1, k2, mu2, ph2):
    """Symmetrised G2 kernel"""
    k, _, _ = addVectors(k1, mu1, ph1, k2, mu2, ph2)
    if np.isclose(k, 0):
        return 0
    else:
        k1pk2 = scalarProduct(k1, mu1, ph1, k2, mu2, ph2)
        F2 = (
            3 / 7
            + 1 / 2 * (1 / k1**2 + 1 / k2**2) * k1pk2
            + 4 / 7 * k1pk2**2 / (k1 * k2) ** 2
        )
        return F2


@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64)",
    fastmath=True,
)
def _F3_T1_symetrised_23(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """in this combination the IR divergence can be resumed
    symmetrised in the second and third argument
    """
    k23, mu23, ph23 = addVectors(k2, mu2, ph2, k3, mu3, ph3)
    if np.isclose(k23, 0):
        return 0
    else:
        F2_23 = vF2(k2, mu2, ph2, k3, mu3, ph3)
        G2_23 = vG2(k2, mu2, ph2, k3, mu3, ph3)
        a23 = alpha(k1, mu1, ph1, k23, mu23, ph23)
        b23 = beta(k1, mu1, ph1, k23, mu23, ph23)

        result = 1 / 18 * (7 * a23 * F2_23 + 2 * b23 * G2_23)
        return result


@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64)",
    fastmath=True,
)
def _F3_T2_symetrised_12(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """in this combination the IR divergence can be resumed
    symmetrised in the first and second argument
    """
    k12, mu12, ph12 = addVectors(k1, mu1, ph1, k2, mu2, ph2)
    if np.isclose(k12, 0):
        return 0
    else:
        G2_12 = vG2(k1, mu1, ph1, k2, mu2, ph2)
        a12 = alpha(k12, mu12, ph12, k3, mu3, ph3)
        b12 = beta(k12, mu12, ph12, k3, mu3, ph3)

        return G2_12 / 18 * (7 * a12 + 2 * b12)


@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64)",
    fastmath=True,
)
def vF3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """Computes the F3 mode coupling kernel"""
    F3 = _F3_T1_symetrised_23(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    F3 += _F3_T1_symetrised_23(k2, mu2, ph2, k3, mu3, ph3, k1, mu1, ph1)
    F3 += _F3_T1_symetrised_23(k3, mu3, ph3, k1, mu1, ph1, k2, mu2, ph2)
    F3 += _F3_T2_symetrised_12(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    F3 += _F3_T2_symetrised_12(k2, mu2, ph2, k3, mu3, ph3, k1, mu1, ph1)
    F3 += _F3_T2_symetrised_12(k3, mu3, ph3, k1, mu1, ph1, k2, mu2, ph2)
    return F3 / 3


@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64)",
    fastmath=True,
)
def _G3_T1_symetrised_23(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """in this combination the IR divergence can be resumed
    symmetrised in the second and third argument
    """
    k23, mu23, ph23 = addVectors(k2, mu2, ph2, k3, mu3, ph3)
    if np.isclose(k23, 0):
        return 0
    else:
        F2_23 = vF2(k2, mu2, ph2, k3, mu3, ph3)
        G2_23 = vG2(k2, mu2, ph2, k3, mu3, ph3)
        a23 = alpha(k1, mu1, ph1, k23, mu23, ph23)
        b23 = beta(k1, mu1, ph1, k23, mu23, ph23)

        result = 1 / 18 * (3 * a23 * F2_23 + 6 * b23 * G2_23)
        return result


@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64)",
    fastmath=True,
)
def _G3_T2_symetrised_12(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """in this combination the IR divergence can be resumed
    symmetrised in the first and second argument
    """
    k12, mu12, ph12 = addVectors(k1, mu1, ph1, k2, mu2, ph2)
    if np.isclose(k12, 0):
        return 0
    else:
        G2_12 = vG2(k1, mu1, ph1, k2, mu2, ph2)
        a12 = alpha(k12, mu12, ph12, k3, mu3, ph3)
        b12 = beta(k12, mu12, ph12, k3, mu3, ph3)

        return G2_12 / 18 * (3 * a12 + 6 * b12)


@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64)",
    fastmath=True,
)
def vG3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """Computes the G3 mode coupling kernel"""
    G3 = _G3_T1_symetrised_23(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    G3 += _G3_T1_symetrised_23(k2, mu2, ph2, k3, mu3, ph3, k1, mu1, ph1)
    G3 += _G3_T1_symetrised_23(k3, mu3, ph3, k1, mu1, ph1, k2, mu2, ph2)
    G3 += _G3_T2_symetrised_12(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    G3 += _G3_T2_symetrised_12(k2, mu2, ph2, k3, mu3, ph3, k1, mu1, ph1)
    G3 += _G3_T2_symetrised_12(k3, mu3, ph3, k1, mu1, ph1, k2, mu2, ph2)
    return G3 / 3


###############
# RSD Kernels #
###############
# These functions are for the approximation that the mean bias is roughly independent
# from the k-dependent shape inside these integrals


@njit(
    "(float64, float64," + "float64, float64, float64)",
    fastmath=True,
)
def vZ1(mb1, f, k1, mu1, ph1):
    """Kaiser term RSD function"""
    return mb1 + f * mu1**2  # for now no b k^2 term


@njit(
    "(float64, float64, float64, float64,"
    + "float64, float64, float64,"
    + "float64, float64, float64)",
    fastmath=True,
)
def vZ2(mb1, mb2, mbG2, f, k1, mu1, ph1, k2, mu2, ph2):
    """Second order RSD mode coupling kernel"""
    k12, mu12, ph12 = addVectors(k1, mu1, ph1, k2, mu2, ph2)
    z2 = mb2 / 2
    z2 += mb1 * vF2(k1, mu1, ph1, k2, mu2, ph2)
    z2 += mbG2 * Galileon2(k1, mu1, ph1, k2, mu2, ph2)
    z2 += f * mu12 * k12 / 2 * mb1 * (mu1 / k1 + mu2 / k2)
    z2 += (f * mu12 * k12) ** 2 / 2 * mu1 / k1 * mu2 / k2
    z2 += f * mu12**2 * vG2(k1, mu1, ph1, k2, mu2, ph2)
    return z2


@njit(
    "(float64, float64, float64,"
    + "float64, float64, float64, float64,"
    + "float64,"
    + "float64, float64, float64,"
    + "float64, float64, float64,"
    + "float64, float64, float64)",
    fastmath=True,
)
def _Z3_symetrised_23(
    mb1, mb2, mbG2, mb3, mbdG2, mbG3, mbDG2, f, k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3
):
    """Third order RSD mode coupling kernel symmetrized in the second and third argument
    possible divergences in k23 are captured by the mode coupling kernels in this way
    """
    k23, mu23, ph23 = addVectors(k2, mu2, ph2, k3, mu3, ph3)
    k, mu, ph = addVectors(k1, mu1, ph1, k23, mu23, ph23)

    z3 = mb1 * vF3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    z3 += mb2 * vF2(k2, mu2, ph2, k3, mu3, ph3)
    z3 += mb3 / 6
    z3 += mbdG2 * Galileon2(k2, mu2, ph2, k3, mu3, ph3)
    z3 += (
        2
        * mbG2
        * Galileon2(k1, mu1, ph1, k23, mu23, ph23)
        * vF2(k2, mu2, ph2, k3, mu3, ph3)
    )
    z3 += mbG3 * Galileon3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    z3 += (
        mbDG2
        * Galileon2(k1, mu1, ph1, k23, mu23, ph23)
        * (vF2(k2, mu2, ph2, k3, mu3, ph3) - vG2(k2, mu2, ph2, k3, mu3, ph3))
    )
    z3 += f * mu**2 * vG3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    z3 += (
        f
        * k
        * mu
        * mu1
        / k1
        * (
            mb1 * vF2(k2, mu2, ph2, k3, mu3, ph3)
            + mb2 / 2
            + mbG2 * Galileon2(k2, mu2, ph2, k3, mu3, ph3)
        )
    )
    z3 += (f * k * mu) ** 2 / 2 * mb1 * mu2 / k2 * mu3 / k3
    z3 += (f * k * mu) ** 3 / 6 * mu1 / k1 * mu2 / k2 * mu3 / k3
    z3 += f * k * mu * mb1 * mu23 / k23 * vG2(k2, mu2, ph2, k3, mu3, ph3)
    z3 += (f * k * mu) ** 2 * mu1 / k1 * mu23 / k23 * vG2(k2, mu2, ph2, k3, mu3, ph3)
    return z3


@njit(
    "(float64, float64, float64,"
    + "float64, float64, float64, float64,"
    + "float64,"
    + "float64, float64, float64,"
    + "float64, float64, float64,"
    + "float64, float64, float64)",
    fastmath=True,
)
def vZ3(
    mb1, mb2, mb3, mbG2, mbdG2, mbG3, mbDG2, f, k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3
):
    Z3 = _Z3_symetrised_23(
        mb1,
        mb2,
        mb3,
        mbG2,
        mbdG2,
        mbG3,
        mbDG2,
        f,
        k1,
        mu1,
        ph1,
        k2,
        mu2,
        ph2,
        k3,
        mu3,
        ph3,
    )
    Z3 += _Z3_symetrised_23(
        mb1,
        mb2,
        mb3,
        mbG2,
        mbdG2,
        mbG3,
        mbDG2,
        f,
        k2,
        mu2,
        ph2,
        k3,
        mu3,
        ph3,
        k1,
        mu1,
        ph1,
    )
    Z3 += _Z3_symetrised_23(
        mb1,
        mb2,
        mb3,
        mbG2,
        mbdG2,
        mbG3,
        mbDG2,
        f,
        k3,
        mu3,
        ph3,
        k1,
        mu1,
        ph1,
        k2,
        mu2,
        ph2,
    )
    Z3 *= 1 / 3
    return Z3


#######################
# N-point correlators #
#######################


@njit(
    "(float64,float64, float64, "
    + "float64[::1], float64[::1], float64, "
    + "float64[::1], float64[::1])",
    fastmath=True,
)
def PowerSpectrumLO(Lmb1_1, Lmb1_2, f, k1, mu1, ph1, kgrid, Pgrid):
    """Redshift space halo power spectrum

    Computes the clustering part of the redshift space halo auto power spectrum.
    Computations are done on a (k, mu) grid. Phi does not do anything because of rotational invariances.
    Can pass different mean biases if they are weighted by different luminosities (13 or 22)
    """
    P = np.exp(linear_interpolate(np.log(kgrid), np.log(Pgrid), np.log(k1)))
    lk, lmu = len(k1), len(mu1)
    Z11 = np.empty((lk, lmu))
    Z12 = np.empty((lk, lmu))
    for ik, ki in enumerate(k1):
        for imu, mui in enumerate(mu1):
            Z11[ik, imu] = vZ1(Lmb1_1, f, ki, mui, ph1)
            Z12[ik, imu] = vZ1(Lmb1_2, f, ki, mui, ph1)
    return Z11 * Z12 * P[:, None]


@njit(
    "(float64, float64, float64, "
    + "float64, float64, float64, "
    + "float64, "
    + "float64, float64, float64, "
    + "float64, float64, float64, "
    + "float64[::1], float64[::1])",
    fastmath=True,
)
def BispectrumLO(
    Lmb1_1,
    Lmb2_1,
    LmbG2_1,
    Lmb1_2,
    Lmb2_2,
    LmbG2_2,
    f,
    k1,
    mu1,
    ph1,
    k2,
    mu2,
    ph2,
    kgrid,
    Pgrid,
):
    """Redshift space halo trispectrum

    Computes the 3 halo clustering contribution redshift space power spectrum.
    All computations are done on a single point. The third vector is computed from the first two.
    Can pass different mean biases if they are weighted by different luminosities (112).
    """
    # compute missing wave vector
    k3, mu3, ph3 = addVectors(k1, -mu1, ph1 + np.pi, k2, -mu2, ph2 + np.pi)
    # Obtain the Power Spectra w/ power-law extrapolation
    vk = np.array([k1, k2, k3])
    vP = np.exp(linear_interpolate(np.log(kgrid), np.log(Pgrid), np.log(vk)))

    # Compute over all permutations of F2 diagrams
    T = 0
    if not np.isclose(k1, 0) and not np.isclose(k2, 0):
        Z11 = vZ1(Lmb1_1, f, k1, mu1, ph1)
        Z12 = vZ1(Lmb1_1, f, k2, mu2, ph2)
        Z21 = vZ2(
            Lmb1_2, Lmb2_2, LmbG2_2, f, k1, -mu1, ph1 + np.pi, k2, -mu2, ph2 + np.pi
        )
        T += 2 * vP[0] * vP[1] * Z11 * Z12 * Z21
    if not np.isclose(k1, 0) and not np.isclose(k3, 0):
        Z11 = vZ1(Lmb1_2, f, k3, mu3, ph3)
        Z12 = vZ1(Lmb1_1, f, k1, mu1, ph1)
        Z21 = vZ2(
            Lmb1_1, Lmb2_1, LmbG2_1, f, k3, -mu3, ph3 + np.pi, k1, -mu1, ph1 + np.pi
        )
        T += 2 * vP[2] * vP[0] * Z11 * Z12 * Z21
    if not np.isclose(k2, 0) and not np.isclose(k3, 0):
        Z11 = vZ1(Lmb1_1, f, k2, mu2, ph2)
        Z12 = vZ1(Lmb1_2, f, k3, mu3, ph3)
        Z21 = vZ2(
            Lmb1_1, Lmb2_1, LmbG2_1, f, k2, -mu2, ph2 + np.pi, k3, -mu3, ph3 + np.pi
        )
        T += 2 * vP[1] * vP[2] * Z11 * Z12 * Z21
    return T


@njit(
    "(float64, "
    + "float64, float64,"
    + "float64, float64, float64, float64, "
    + "float64, "
    + "float64, float64, float64, "
    + "float64, float64, float64, "
    + "float64, float64, float64, "
    + "float64[::1], float64[::1])",
    fastmath=True,
)
def TrispectrumL0(
    Lmb1,
    Lmb2,
    LmbG2,
    Lmb3,
    LmbdG2,
    LmbG3,
    LmbDG2,
    f,
    k1,
    mu1,
    ph1,
    k2,
    mu2,
    ph2,
    k3,
    mu3,
    ph3,
    kgrid,
    Pgrid,
):
    """Redshift space halo Trispectrum

    Computes the 4 halo clustering contribution redshift space power spectrum.
    All computations are done on a single point. The forth vector is computed from the first three.
    The different biases are assumed to be weighed by same powers of luminosity (1111).
    """
    # Compute missing vector
    k12, mu12, ph12 = addVectors(k1, mu1, ph1, k2, mu2, ph2)
    k4, mu4, ph4 = addVectors(k12, -mu12, ph12 + np.pi, k3, -mu3, ph3 + np.pi)
    # Compute coordinates of added wave vectors
    k13, mu13, ph13 = addVectors(k1, mu1, ph1, k3, mu3, ph3)
    k14, mu14, ph14 = addVectors(k1, mu1, ph1, k4, mu4, ph4)
    k23, mu23, ph23 = addVectors(k2, mu2, ph2, k3, mu3, ph3)
    k24, mu24, ph24 = addVectors(k2, mu2, ph2, k4, mu4, ph4)
    k34, mu34, ph34 = addVectors(k3, mu3, ph3, k4, mu4, ph4)

    # Obtain the Power Spectra
    vk = np.array([k1, k2, k3, k4, k12, k13, k14])
    # Power-law Extrapolation
    vlogP = linear_interpolate(np.log(kgrid), np.log(Pgrid), np.log(vk))
    vP = np.exp(vlogP)

    T1 = 0
    # Compute over all permutations of the 1122 diagrams
    if not np.isclose(k12, 0):
        Z11 = vZ1(Lmb1, f, k1, mu1, ph1)
        Z12 = vZ1(Lmb1, f, k2, mu2, ph2)
        Z13 = vZ1(Lmb1, f, k3, mu3, ph3)
        Z14 = vZ1(Lmb1, f, k4, mu4, ph4)

        Z21 = vZ2(Lmb1, Lmb2, LmbG2, f, k1, -mu1, ph1 + np.pi, k12, mu12, ph12)
        Z22 = vZ2(Lmb1, Lmb2, LmbG2, f, k2, -mu2, ph2 + np.pi, k12, mu12, ph12)
        Z23 = vZ2(Lmb1, Lmb2, LmbG2, f, k3, -mu3, ph3 + np.pi, k34, mu34, ph34)
        Z24 = vZ2(Lmb1, Lmb2, LmbG2, f, k4, -mu4, ph4 + np.pi, k34, mu34, ph34)

        T1 += 4 * Z11 * Z13 * Z21 * Z23 * vP[0] * vP[2] * vP[4]
        T1 += 4 * Z11 * Z14 * Z21 * Z24 * vP[0] * vP[3] * vP[4]
        T1 += 4 * Z12 * Z13 * Z22 * Z23 * vP[1] * vP[2] * vP[4]
        T1 += 4 * Z12 * Z14 * Z22 * Z24 * vP[1] * vP[3] * vP[4]

    if not np.isclose(k13, 0):
        Z11 = vZ1(Lmb1, f, k1, mu1, ph1)
        Z12 = vZ1(Lmb1, f, k2, mu2, ph2)
        Z12 = vZ1(Lmb1, f, k3, mu3, ph3)
        Z12 = vZ1(Lmb1, f, k4, mu4, ph4)

        Z21 = vZ2(Lmb1, Lmb2, LmbG2, f, k1, -mu1, ph1 + np.pi, k13, mu13, ph13)
        Z22 = vZ2(Lmb1, Lmb2, LmbG2, f, k2, -mu2, ph2 + np.pi, k24, mu24, ph24)
        Z23 = vZ2(Lmb1, Lmb2, LmbG2, f, k3, -mu3, ph3 + np.pi, k13, mu13, ph13)
        Z24 = vZ2(Lmb1, Lmb2, LmbG2, f, k4, -mu4, ph4 + np.pi, k24, mu24, ph24)

        T1 += 4 * Z11 * Z12 * Z21 * Z22 * vP[0] * vP[1] * vP[5]
        T1 += 4 * Z11 * Z14 * Z21 * Z24 * vP[0] * vP[3] * vP[5]
        T1 += 4 * Z12 * Z13 * Z22 * Z23 * vP[1] * vP[2] * vP[5]
        T1 += 4 * Z13 * Z14 * Z23 * Z24 * vP[2] * vP[3] * vP[5]

    if not np.isclose(k14, 0):
        Z11 = vZ1(Lmb1, f, k1, mu1, ph1)
        Z12 = vZ1(Lmb1, f, k2, mu2, ph2)
        Z12 = vZ1(Lmb1, f, k3, mu3, ph3)
        Z12 = vZ1(Lmb1, f, k4, mu4, ph4)

        Z21 = vZ2(Lmb1, Lmb2, LmbG2, f, k1, -mu1, ph1 + np.pi, k14, mu14, ph14)
        Z22 = vZ2(Lmb1, Lmb2, LmbG2, f, k2, -mu2, ph2 + np.pi, k23, mu23, ph23)
        Z23 = vZ2(Lmb1, Lmb2, LmbG2, f, k3, -mu3, ph3 + np.pi, k23, mu23, ph23)
        Z24 = vZ2(Lmb1, Lmb2, LmbG2, f, k4, -mu4, ph4 + np.pi, k14, mu14, ph14)

        T1 += 4 * Z11 * Z12 * Z21 * Z22 * vP[0] * vP[1] * vP[6]
        T1 += 4 * Z11 * Z13 * Z21 * Z23 * vP[0] * vP[2] * vP[6]
        T1 += 4 * Z12 * Z14 * Z22 * Z24 * vP[1] * vP[3] * vP[6]
        T1 += 4 * Z13 * Z14 * Z23 * Z24 * vP[2] * vP[3] * vP[6]

    # Compute over all permutations of the 1113 diagrams
    Z31 = vZ3(
        Lmb1,
        Lmb2,
        LmbG2,
        Lmb3,
        LmbdG2,
        LmbG3,
        LmbDG2,
        f,
        k2,
        mu2,
        ph2,
        k3,
        mu3,
        ph3,
        k4,
        mu4,
        ph4,
    )
    Z32 = vZ3(
        Lmb1,
        Lmb2,
        LmbG2,
        Lmb3,
        LmbdG2,
        LmbG3,
        LmbDG2,
        f,
        k3,
        mu3,
        ph3,
        k4,
        mu4,
        ph4,
        k1,
        mu1,
        ph1,
    )
    Z33 = vZ3(
        Lmb1,
        Lmb2,
        LmbG2,
        Lmb3,
        LmbdG2,
        LmbG3,
        LmbDG2,
        f,
        k4,
        mu4,
        ph4,
        k1,
        mu1,
        ph1,
        k2,
        mu2,
        ph2,
    )
    Z34 = vZ3(
        Lmb1,
        Lmb2,
        LmbG2,
        Lmb3,
        LmbdG2,
        LmbG3,
        LmbDG2,
        f,
        k1,
        mu1,
        ph1,
        k2,
        mu2,
        ph2,
        k3,
        mu3,
        ph3,
    )

    T2 = 6 * Z31 * vP[1] * vP[2] * vP[3]
    T2 += 6 * Z32 * vP[0] * vP[2] * vP[3]
    T2 += 6 * Z33 * vP[0] * vP[1] * vP[3]
    T2 += 6 * Z34 * vP[0] * vP[1] * vP[2]

    if np.isnan(T1) or np.isnan(T2):
        print(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3, k4, mu4, ph4, kgrid, Pgrid)
        raise RuntimeError("NaN encountered", T1, T2)

    return T1 + T2


########################
# Integration routines #
########################


@njit(
    "(float64, float64, float64, float64, "
    + "float64[::1], float64[::1], float64[::1], float64[::1], "
    + "float64[:,:], float64[:,:,:,:], float64[:,:,:,:])"
)
def _integrate_2h(Lmb1, L2mb1, L3mb1, f, xi, w, k, Pgrid, I01, I02, I03):
    """Integrated 2h contributions

    Inputs are the Luminosity weighted linear bias for a linear, quadratic and cubic weight,
    the effective growth rate entering in the RSD contribution,
    Weights and roots of the Legendre polynomials for gauss Legendre integration,
    k and P grid to do interpolation when needed,
    Fourier-transformed luminosity-weight mean halo profile.
    """

    assert len(xi) == len(
        w
    ), "Number of integration points must match number of weights"
    nnodes = len(xi)
    kl = len(k)

    mu = xi
    phi = np.pi * xi

    # obtain necessary Legendre functions
    L0 = legendre_0(mu) * (2 * 0 + 1) / 2
    L2 = legendre_2(mu) * (2 * 2 + 1) / 2
    L4 = legendre_4(mu) * (2 * 4 + 1) / 2

    pseudo_Cov = np.zeros((kl, kl, 3, 3))
    for ik1 in prange(kl):
        for ik2 in prange(ik1, kl):

            mu1_integ0 = np.empty(nnodes)
            mu1_integ2 = np.empty(nnodes)
            mu1_integ4 = np.empty(nnodes)
            for imu1 in range(nnodes):
                mu2_integ = np.empty(nnodes)
                for imu2 in range(nnodes):
                    phi1_integ = np.empty(nnodes)
                    for iphi1 in range(nnodes):
                        phi2_integ = np.empty(nnodes)
                        for iphi2 in range(nnodes):
                            hTs = 0
                            # 22 halos
                            k13, mu13, ph13 = addVectors(
                                k[ik1],
                                mu[imu1],
                                phi[iphi1],
                                k[ik2],
                                mu[imu2],
                                phi[iphi2],
                            )
                            vk, vmu = np.array([k13]), np.array([mu13])
                            P22 = PowerSpectrumLO(
                                L2mb1, L2mb1, f, vk, vmu, ph13, k, Pgrid
                            )
                            hTs += P22[0, 0] * I02[ik1, ik2, imu1, imu2] ** 2

                            k14, mu14, ph14 = addVectors(
                                k[ik1],
                                mu[imu1],
                                phi[iphi1],
                                k[ik2],
                                -mu[imu2],
                                phi[iphi2] + np.pi,
                            )
                            vk, vmu = np.array([k14]), np.array([mu14])
                            P22 = PowerSpectrumLO(
                                L2mb1, L2mb1, f, vk, vmu, ph14, k, Pgrid
                            )
                            hTs += P22[0, 0] * I02[ik1, ik2, imu1, imu2] ** 2

                            # 31 halos
                            vk, vmu = np.array([k[ik1]]), np.array([mu[imu1]])
                            P31 = PowerSpectrumLO(Lmb1, L3mb1, f, vk, vmu, 0, k, Pgrid)
                            hTs += (
                                2
                                * P31[0, 0]
                                * I03[ik2, ik1, imu2, imu1]
                                * I01[ik1, imu1]
                            )

                            vk, vmu = np.array([k[ik2]]), np.array([mu[imu2]])
                            P31 = PowerSpectrumLO(Lmb1, L3mb1, f, vk, vmu, 0, k, Pgrid)
                            hTs += (
                                2
                                * P31[0, 0]
                                * I03[ik1, ik2, imu1, imu2]
                                * I01[ik2, imu2]
                            )

                            phi2_integ[iphi2] = hTs / (4 * np.pi) ** 2
                        phi1_integ[iphi1] = np.sum(phi2_integ * w * np.pi)
                    mu2_integ[imu2] = np.sum(phi1_integ * w * np.pi)
                # integrate over mu2 first
                mu1_integ0[imu1] = np.sum(mu2_integ * L0 * w)
                mu1_integ2[imu1] = np.sum(mu2_integ * L2 * w)
                mu1_integ4[imu1] = np.sum(mu2_integ * L4 * w)
            pseudo_Cov[ik1, ik2, 0, 0] = np.sum(mu1_integ0 * L0 * w)
            pseudo_Cov[ik1, ik2, 0, 1] = np.sum(mu1_integ2 * L0 * w)
            pseudo_Cov[ik1, ik2, 0, 2] = np.sum(mu1_integ4 * L0 * w)
            pseudo_Cov[ik1, ik2, 1, 0] = np.sum(mu1_integ0 * L2 * w)
            pseudo_Cov[ik1, ik2, 1, 1] = np.sum(mu1_integ2 * L2 * w)
            pseudo_Cov[ik1, ik2, 1, 2] = np.sum(mu1_integ4 * L2 * w)
            pseudo_Cov[ik1, ik2, 2, 0] = np.sum(mu1_integ0 * L4 * w)
            pseudo_Cov[ik1, ik2, 2, 1] = np.sum(mu1_integ2 * L4 * w)
            pseudo_Cov[ik1, ik2, 2, 2] = np.sum(mu1_integ4 * L4 * w)

            # use symmetries k1 <-> k2
            pseudo_Cov[ik2, ik1, 0, 0] = pseudo_Cov[ik1, ik2, 0, 0]
            pseudo_Cov[ik2, ik1, 1, 0] = pseudo_Cov[ik1, ik2, 0, 1]
            pseudo_Cov[ik2, ik1, 2, 0] = pseudo_Cov[ik1, ik2, 0, 2]
            pseudo_Cov[ik2, ik1, 0, 1] = pseudo_Cov[ik1, ik2, 1, 0]
            pseudo_Cov[ik2, ik1, 1, 1] = pseudo_Cov[ik1, ik2, 1, 1]
            pseudo_Cov[ik2, ik1, 2, 1] = pseudo_Cov[ik1, ik2, 1, 2]
            pseudo_Cov[ik2, ik1, 0, 2] = pseudo_Cov[ik1, ik2, 2, 0]
            pseudo_Cov[ik2, ik1, 1, 2] = pseudo_Cov[ik1, ik2, 2, 1]
            pseudo_Cov[ik2, ik1, 2, 2] = pseudo_Cov[ik1, ik2, 2, 2]

    return pseudo_Cov


@njit(
    "(float64, float64, float64, float64, float64, float64, float64, "
    + "float64[::1], float64[::1], float64[::1], float64[::1], "
    + "float64[:,:], float64[:,:,:,:])",
    parallel=True,
)
def _integrate_3h(
    Lmb1, Lmb2, LmbG2, L2mb1, L2mb2, L2mbG2, f, xi, w, k, Pgrid, I01, I02
):
    assert len(xi) == len(
        w
    ), "Number of integration points must match number of weights"
    nnodes = len(xi)
    kl = len(k)

    mu = xi
    phi = np.pi * xi

    # obtain necessary Legendre functions
    L0 = legendre_0(mu) * (2 * 0 + 1) / 2
    L2 = legendre_2(mu) * (2 * 2 + 1) / 2
    L4 = legendre_4(mu) * (2 * 4 + 1) / 2

    pseudo_Cov = np.zeros((kl, kl, 3, 3))
    for ik1 in prange(kl):
        for ik2 in prange(ik1, kl):

            mu1_integ0 = np.empty(nnodes)
            mu1_integ2 = np.empty(nnodes)
            mu1_integ4 = np.empty(nnodes)
            for imu1 in range(nnodes):
                mu2_integ = np.empty(nnodes)
                for imu2 in range(nnodes):
                    phi1_integ = np.empty(nnodes)
                    for iphi1 in range(nnodes):
                        phi2_integ = np.empty(nnodes)
                        for iphi2 in range(nnodes):
                            phi2_integ[iphi2] = (
                                BispectrumLO(
                                    Lmb1,
                                    Lmb2,
                                    LmbG2,
                                    L2mb1,
                                    L2mb2,
                                    L2mbG2,
                                    f,
                                    k[ik1],
                                    -mu[imu1],
                                    phi[iphi1] + np.pi,
                                    k[ik2],
                                    -mu[imu2],
                                    phi[iphi2] + np.pi,
                                    k,
                                    Pgrid,
                                )
                                * I02[ik1, ik2, imu1, imu2]
                                * I01[ik1, imu1]
                                * I01[ik2, imu2]
                            )

                            phi2_integ[iphi2] += (
                                BispectrumLO(
                                    Lmb1,
                                    Lmb2,
                                    LmbG2,
                                    L2mb1,
                                    L2mb2,
                                    L2mbG2,
                                    f,
                                    k[ik1],
                                    -mu[imu1],
                                    phi[iphi1] + np.pi,
                                    k[ik2],
                                    mu[imu2],
                                    phi[iphi2],
                                    k,
                                    Pgrid,
                                )
                                * I02[ik1, ik2, imu1, imu2]
                                * I01[ik1, imu1]
                                * I01[ik2, imu2]
                            )

                            phi2_integ[iphi2] += (
                                BispectrumLO(
                                    Lmb1,
                                    Lmb2,
                                    LmbG2,
                                    L2mb1,
                                    L2mb2,
                                    L2mbG2,
                                    f,
                                    k[ik1],
                                    mu[imu1],
                                    phi[iphi1],
                                    k[ik2],
                                    -mu[imu2],
                                    phi[iphi2] + np.pi,
                                    k,
                                    Pgrid,
                                )
                                * I02[ik1, ik2, imu1, imu2]
                                * I01[ik1, imu1]
                                * I01[ik2, imu2]
                            )

                            phi2_integ[iphi2] += (
                                BispectrumLO(
                                    Lmb1,
                                    Lmb2,
                                    LmbG2,
                                    L2mb1,
                                    L2mb2,
                                    L2mbG2,
                                    f,
                                    k[ik1],
                                    mu[imu1],
                                    phi[iphi1],
                                    k[ik2],
                                    mu[imu2],
                                    phi[iphi2],
                                    k,
                                    Pgrid,
                                )
                                * I02[ik1, ik2, imu1, imu2]
                                * I01[ik1, imu1]
                                * I01[ik2, imu2]
                            )

                        phi2_integ = phi2_integ / (4 * np.pi) ** 2
                        phi1_integ[iphi1] = np.sum(phi2_integ * w * np.pi)
                    mu2_integ[imu2] = np.sum(phi1_integ * w * np.pi)
                # integrate over mu2 first
                mu1_integ0[imu1] = np.sum(mu2_integ * L0 * w)
                mu1_integ2[imu1] = np.sum(mu2_integ * L2 * w)
                mu1_integ4[imu1] = np.sum(mu2_integ * L4 * w)
            pseudo_Cov[ik1, ik2, 0, 0] = np.sum(mu1_integ0 * L0 * w)
            pseudo_Cov[ik1, ik2, 0, 1] = np.sum(mu1_integ2 * L0 * w)
            pseudo_Cov[ik1, ik2, 0, 2] = np.sum(mu1_integ4 * L0 * w)
            pseudo_Cov[ik1, ik2, 1, 0] = np.sum(mu1_integ0 * L2 * w)
            pseudo_Cov[ik1, ik2, 1, 1] = np.sum(mu1_integ2 * L2 * w)
            pseudo_Cov[ik1, ik2, 1, 2] = np.sum(mu1_integ4 * L2 * w)
            pseudo_Cov[ik1, ik2, 2, 0] = np.sum(mu1_integ0 * L4 * w)
            pseudo_Cov[ik1, ik2, 2, 1] = np.sum(mu1_integ2 * L4 * w)
            pseudo_Cov[ik1, ik2, 2, 2] = np.sum(mu1_integ4 * L4 * w)

            # use symmetries k1 <-> k2
            pseudo_Cov[ik2, ik1, 0, 0] = pseudo_Cov[ik1, ik2, 0, 0]
            pseudo_Cov[ik2, ik1, 1, 0] = pseudo_Cov[ik1, ik2, 0, 1]
            pseudo_Cov[ik2, ik1, 2, 0] = pseudo_Cov[ik1, ik2, 0, 2]
            pseudo_Cov[ik2, ik1, 0, 1] = pseudo_Cov[ik1, ik2, 1, 0]
            pseudo_Cov[ik2, ik1, 1, 1] = pseudo_Cov[ik1, ik2, 1, 1]
            pseudo_Cov[ik2, ik1, 2, 1] = pseudo_Cov[ik1, ik2, 1, 2]
            pseudo_Cov[ik2, ik1, 0, 2] = pseudo_Cov[ik1, ik2, 2, 0]
            pseudo_Cov[ik2, ik1, 1, 2] = pseudo_Cov[ik1, ik2, 2, 1]
            pseudo_Cov[ik2, ik1, 2, 2] = pseudo_Cov[ik1, ik2, 2, 2]

    return pseudo_Cov


@njit(
    "(float64, float64, float64, float64, float64, float64, float64, float64, "
    + "float64[::1], float64[::1], float64[::1], float64[::1], float64[:,:])",
    parallel=True,
)
def _integrate_4h(
    Lmb1, Lmb2, LmbG2, Lmb3, LmbdG2, LmbG3, LmbDG2, f, xi, w, k, Pgrid, I01
):
    assert len(xi) == len(
        w
    ), "Number of integration points must match number of weights"
    nnodes = len(xi)
    kl = len(k)

    mu = xi
    phi = np.pi * xi

    # obtain necessary Legendre functions
    L0 = legendre_0(mu) * (2 * 0 + 1) / 2
    L2 = legendre_2(mu) * (2 * 2 + 1) / 2
    L4 = legendre_4(mu) * (2 * 4 + 1) / 2

    pseudo_Cov = np.zeros((kl, kl, 3, 3))
    for ik1 in prange(kl):
        for ik2 in prange(ik1, kl):

            mu1_integ0 = np.empty(nnodes)
            mu1_integ2 = np.empty(nnodes)
            mu1_integ4 = np.empty(nnodes)
            for imu1 in range(nnodes):
                mu2_integ = np.empty(nnodes)
                for imu2 in range(nnodes):
                    phi1_integ = np.empty(nnodes)
                    for iphi1 in range(nnodes):
                        phi2_integ = np.empty(nnodes)
                        for iphi2 in range(nnodes):
                            phi2_integ[iphi2] = (
                                TrispectrumL0(
                                    Lmb1,
                                    Lmb2,
                                    LmbG2,
                                    Lmb3,
                                    LmbdG2,
                                    LmbG3,
                                    LmbDG2,
                                    f,
                                    k[ik1],
                                    mu[imu1],
                                    phi[iphi1],
                                    k[ik2],
                                    mu[imu2],
                                    phi[iphi2],
                                    k[ik1],
                                    -mu[imu1],
                                    phi[iphi1] + np.pi,
                                    k,
                                    Pgrid,
                                )
                                / (4 * np.pi) ** 2
                                * I01[ik1, imu1] ** 2
                                * I01[ik2, imu2] ** 2
                            )
                        phi1_integ[iphi1] = np.sum(phi2_integ * w * np.pi)
                    mu2_integ[imu2] = np.sum(phi1_integ * w * np.pi)
                # integrate over mu2 first
                mu1_integ0[imu1] = np.sum(mu2_integ * L0 * w)
                mu1_integ2[imu1] = np.sum(mu2_integ * L2 * w)
                mu1_integ4[imu1] = np.sum(mu2_integ * L4 * w)
            pseudo_Cov[ik1, ik2, 0, 0] = np.sum(mu1_integ0 * L0 * w)
            pseudo_Cov[ik1, ik2, 0, 1] = np.sum(mu1_integ2 * L0 * w)
            pseudo_Cov[ik1, ik2, 0, 2] = np.sum(mu1_integ4 * L0 * w)
            pseudo_Cov[ik1, ik2, 1, 0] = np.sum(mu1_integ0 * L2 * w)
            pseudo_Cov[ik1, ik2, 1, 1] = np.sum(mu1_integ2 * L2 * w)
            pseudo_Cov[ik1, ik2, 1, 2] = np.sum(mu1_integ4 * L2 * w)
            pseudo_Cov[ik1, ik2, 2, 0] = np.sum(mu1_integ0 * L4 * w)
            pseudo_Cov[ik1, ik2, 2, 1] = np.sum(mu1_integ2 * L4 * w)
            pseudo_Cov[ik1, ik2, 2, 2] = np.sum(mu1_integ4 * L4 * w)

            # use symmetry  k1 <-> k2
            pseudo_Cov[ik2, ik1, 0, 0] = pseudo_Cov[ik1, ik2, 0, 0]
            pseudo_Cov[ik2, ik1, 1, 0] = pseudo_Cov[ik1, ik2, 0, 1]
            pseudo_Cov[ik2, ik1, 2, 0] = pseudo_Cov[ik1, ik2, 0, 2]
            pseudo_Cov[ik2, ik1, 0, 1] = pseudo_Cov[ik1, ik2, 1, 0]
            pseudo_Cov[ik2, ik1, 1, 1] = pseudo_Cov[ik1, ik2, 1, 1]
            pseudo_Cov[ik2, ik1, 2, 1] = pseudo_Cov[ik1, ik2, 1, 2]
            pseudo_Cov[ik2, ik1, 0, 2] = pseudo_Cov[ik1, ik2, 2, 0]
            pseudo_Cov[ik2, ik1, 1, 2] = pseudo_Cov[ik1, ik2, 2, 1]
            pseudo_Cov[ik2, ik1, 2, 2] = pseudo_Cov[ik1, ik2, 2, 2]

    return pseudo_Cov
