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
def _F3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """Method to compute the un-symmetrised F3

    Uses the reccusrion relation in [0112551]
    """
    # Prefactor
    pf = 1 / 18

    # First Term
    q1, muq1, phq1 = k1, mu1, ph1
    q2, muq2, phq2 = addVectors(k2, mu2, ph2, k3, mu3, ph3)
    T11 = 7 * alpha(q1, muq1, phq1, q2, muq2, phq2) * vF2(k2, mu2, ph2, k3, mu3, ph3)
    T12 = 0
    if not np.isclose(q2, 0.0):
        T12 = (
            2
            * beta(q1, muq1, phq1, q2, muq2, phq2)  # divergeces with 1/q2
            * vG2(k2, mu2, ph2, k3, mu3, ph3)  # vanishes with q2^2
        )
    T1 = T11 + T12

    # Second Term
    q1, muq1, phq1 = addVectors(k1, mu1, ph1, k2, mu2, ph2)
    q2, muq2, phq2 = k3, mu3, ph3
    T2 = 0
    if not np.isclose(q1, 0.0):
        T2 = vG2(k1, mu1, ph1, k2, mu2, ph2) * (  # vanishes with q1^2
            7 * alpha(q1, muq1, phq1, q2, muq2, phq2)  # divergeces with 1/q1
            + 2 * beta(q1, muq1, phq1, q2, muq2, phq2)  # divergeces with 1/q1
        )
    return pf * (T1 + T2)


@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64)",
    fastmath=True,
)
def vF3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """Method to compute the F3 mode coupling kernel"""
    sF3 = (
        _F3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
        + _F3(k1, mu1, ph1, k3, mu3, ph3, k2, mu2, ph2)
        + _F3(k2, mu2, ph2, k1, mu1, ph1, k3, mu3, ph3)
        + _F3(k2, mu2, ph2, k3, mu3, ph3, k1, mu1, ph1)
        + _F3(k3, mu3, ph3, k1, mu1, ph1, k2, mu2, ph2)
        + _F3(k3, mu3, ph3, k2, mu2, ph2, k1, mu1, ph1)
    )
    return sF3 / 6


@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64)",
    fastmath=True,
)
def _G3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """Method to compute the un-symmetrised G3

    Uses the reccusrion relation in [0112551]
    """
    # Prefactor
    pf = 1 / 18

    # First Term
    q1, muq1, phq1 = k1, mu1, ph1
    q2, muq2, phq2 = addVectors(k2, mu2, ph2, k3, mu3, ph3)
    T11 = 3 * alpha(q1, muq1, phq1, q2, muq2, phq2) * vF2(k2, mu2, ph2, k3, mu3, ph3)
    T12 = 0
    if not np.isclose(q2, 0.0):
        T12 = (
            6
            * beta(q1, muq1, phq1, q2, muq2, phq2)  # divergeces with 1/q2
            * vG2(k2, mu2, ph2, k3, mu3, ph3)  # vanishes with q2^2
        )
    T1 = T11 + T12

    # Second Term
    q1, muq1, phq1 = addVectors(k1, mu1, ph1, k2, mu2, ph2)
    q2, muq2, phq2 = k3, mu3, ph3
    T2 = 0
    if not np.isclose(q1, 0.0):
        T2 = vG2(k1, mu1, ph1, k2, mu2, ph2) * (  # vanishes with q1^2
            3 * alpha(q1, muq1, phq1, q2, muq2, phq2)  # divergeces with 1/q1
            + 6 * beta(q1, muq1, phq1, q2, muq2, phq2)  # divergeces with 1/q1
        )
    return pf * (T1 + T2)


@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64)",
    fastmath=True,
)
def vG3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """Method to compute the G3 mode coupling kernel"""
    sF3 = (
        _G3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
        + _G3(k1, mu1, ph1, k3, mu3, ph3, k2, mu2, ph2)
        + _G3(k2, mu2, ph2, k1, mu1, ph1, k3, mu3, ph3)
        + _G3(k2, mu2, ph2, k3, mu3, ph3, k1, mu1, ph1)
        + _G3(k3, mu3, ph3, k1, mu1, ph1, k2, mu2, ph2)
        + _G3(k3, mu3, ph3, k2, mu2, ph2, k1, mu1, ph1)
    )
    return sF3 / 6


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

    vparr = f * k * mu

    z3 = mb1 * vF3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    z3 += mb3 / 6
    z3 += mbdG2 * Galileon2(k2, mu2, ph2, k3, mu3, ph3)
    z3 += mbG3 * Galileon3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    z3 += f * mu**2 * vG3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    z3 += (
        vparr
        * mu1
        / k1
        * (
            mb1 * vF2(k2, mu2, ph2, k3, mu3, ph3)
            + mb2 / 2
            + mbG2 * Galileon2(k2, mu2, ph2, k3, mu3, ph3)
        )
    )
    z3 += vparr**2 / 2 * mb1 * mu2 / k2 * mu3 / k3
    z3 += vparr**3 / 6 * mu1 / k1 * mu2 / k2 * mu3 / k3
    if not np.isclose(k23, 0):
        z3 += mb2 * vF2(k2, mu2, ph2, k3, mu3, ph3)
        z3 += (
            2
            * mbG2
            * Galileon2(k1, mu1, ph1, k23, mu23, ph23)
            * vF2(k2, mu2, ph2, k3, mu3, ph3)
        )
        z3 += (
            2
            * mbDG2
            * Galileon2(k1, mu1, ph1, k23, mu23, ph23)
            * (vF2(k2, mu2, ph2, k3, mu3, ph3) - vG2(k2, mu2, ph2, k3, mu3, ph3))
        )
        z3 += vparr * mb1 * mu23 / k23 * vG2(k2, mu2, ph2, k3, mu3, ph3)
        z3 += vparr**2 * mu1 / k1 * mu23 / k23 * vG2(k2, mu2, ph2, k3, mu3, ph3)
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
    mb1, mb2, mbG2, mb3, mbdG2, mbG3, mbDG2, f, k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3
):
    bias = (mb1, mb2, mbG2, mb3, mbdG2, mbG3, mbDG2, f,)
    Z3 = _Z3_symetrised_23(
        *bias,
        k1, mu1, ph1,
        k2, mu2, ph2,
        k3, mu3, ph3,
    )
    Z3 += _Z3_symetrised_23(
        *bias,
        k3, mu3, ph3,
        k1, mu1, ph1,
        k2, mu2, ph2,
    )
    Z3 += _Z3_symetrised_23(
        *bias,
        k2, mu2, ph2,
        k3, mu3, ph3,
        k1, mu1, ph1,
    )
    Z3 *= 1 / 3
    return Z3


#######################
# N-point correlators #
#######################


@njit(
    "(float64,float64, float64, "
    + "float64, float64, float64, "
    + "float64[::1], float64[::1])",
)
def PowerSpectrumLO(Lmb1_1, Lmb1_2, f, k1, mu1, ph1, kgrid, Pgrid):
    """Redshift space halo power spectrum

    Computes the clustering part of the redshift space halo auto power spectrum.
    Can pass different mean biases if they are weighted by different luminosities (13 or 22)
    """
    vk = np.array([k1])
    P = np.exp(linear_interpolate(np.log(kgrid), np.log(Pgrid), np.log(vk)))[0]

    Z11 = vZ1(Lmb1_1, f, k1, mu1, ph1)
    Z12 = vZ1(Lmb1_2, f, k1, mu1, ph1)
    return Z11 * Z12 * P


@njit(
    "(float64, float64, float64, "
    + "float64, float64, float64, "
    + "float64, "
    + "float64, float64, float64, "
    + "float64, float64, float64, "
    + "float64[::1], float64[::1])",
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

    Z11 = vZ1(Lmb1, f, k1, mu1, ph1)
    Z12 = vZ1(Lmb1, f, k2, mu2, ph2)
    Z13 = vZ1(Lmb1, f, k3, mu3, ph3)
    Z14 = vZ1(Lmb1, f, k4, mu4, ph4)

    T1 = 0
    # Compute over all permutations of the 1122 diagrams
    if not np.isclose(k12, 0):
        Z21 = vZ2(Lmb1, Lmb2, LmbG2, f, k1, -mu1, ph1 + np.pi, k12, mu12, ph12)
        Z22 = vZ2(Lmb1, Lmb2, LmbG2, f, k2, -mu2, ph2 + np.pi, k12, mu12, ph12)
        Z23 = vZ2(Lmb1, Lmb2, LmbG2, f, k3, -mu3, ph3 + np.pi, k34, mu34, ph34)
        Z24 = vZ2(Lmb1, Lmb2, LmbG2, f, k4, -mu4, ph4 + np.pi, k34, mu34, ph34)

        T1 += 4 * Z11 * Z13 * Z21 * Z23 * vP[0] * vP[2] * vP[4]
        T1 += 4 * Z11 * Z14 * Z21 * Z24 * vP[0] * vP[3] * vP[4]
        T1 += 4 * Z12 * Z13 * Z22 * Z23 * vP[1] * vP[2] * vP[4]
        T1 += 4 * Z12 * Z14 * Z22 * Z24 * vP[1] * vP[3] * vP[4]

    if not np.isclose(k13, 0):
        Z21 = vZ2(Lmb1, Lmb2, LmbG2, f, k1, -mu1, ph1 + np.pi, k13, mu13, ph13)
        Z22 = vZ2(Lmb1, Lmb2, LmbG2, f, k2, -mu2, ph2 + np.pi, k24, mu24, ph24)
        Z23 = vZ2(Lmb1, Lmb2, LmbG2, f, k3, -mu3, ph3 + np.pi, k13, mu13, ph13)
        Z24 = vZ2(Lmb1, Lmb2, LmbG2, f, k4, -mu4, ph4 + np.pi, k24, mu24, ph24)

        T1 += 4 * Z11 * Z12 * Z21 * Z22 * vP[0] * vP[1] * vP[5]
        T1 += 4 * Z11 * Z14 * Z21 * Z24 * vP[0] * vP[3] * vP[5]
        T1 += 4 * Z12 * Z13 * Z22 * Z23 * vP[1] * vP[2] * vP[5]
        T1 += 4 * Z13 * Z14 * Z23 * Z24 * vP[2] * vP[3] * vP[5]

    if not np.isclose(k14, 0):
        Z21 = vZ2(Lmb1, Lmb2, LmbG2, f, k1, -mu1, ph1 + np.pi, k14, mu14, ph14)
        Z22 = vZ2(Lmb1, Lmb2, LmbG2, f, k2, -mu2, ph2 + np.pi, k23, mu23, ph23)
        Z23 = vZ2(Lmb1, Lmb2, LmbG2, f, k3, -mu3, ph3 + np.pi, k23, mu23, ph23)
        Z24 = vZ2(Lmb1, Lmb2, LmbG2, f, k4, -mu4, ph4 + np.pi, k14, mu14, ph14)

        T1 += 4 * Z11 * Z12 * Z21 * Z22 * vP[0] * vP[1] * vP[6]
        T1 += 4 * Z11 * Z13 * Z21 * Z23 * vP[0] * vP[2] * vP[6]
        T1 += 4 * Z12 * Z14 * Z22 * Z24 * vP[1] * vP[3] * vP[6]
        T1 += 4 * Z13 * Z14 * Z23 * Z24 * vP[2] * vP[3] * vP[6]

    T2 = 0
    # Compute over all permutations of the 1113 diagrams
    if not any_close_to_zero([k2, k3, k4]):
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
        T2 += 6 * Z31 * vP[1] * vP[2] * vP[3]

    if not any_close_to_zero([k3, k4, k1]):
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
        T2 += 6 * Z32 * vP[0] * vP[2] * vP[3]

    if not any_close_to_zero([k4, k1, k2]):
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
        T2 += 6 * Z33 * vP[0] * vP[1] * vP[3]

    if not any_close_to_zero([k1, k2, k3]):
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
        T2 += 6 * Z34 * vP[0] * vP[1] * vP[2]

    if np.isnan(T1) or np.isnan(T2):
        print(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3, k4, mu4, ph4, kgrid, Pgrid)
        raise RuntimeError("NaN encountered", T1, T2)

    return T1 + T2


@njit(
    "(float64, float64, float64, float64, float64, float64, float64, float64, "
    + "float64, float64, float64, "
    + "float64, float64, float64, "
    + "float64[::1], float64[::1])",
)
def collapsed_Trispectrum_LO(
    Lmb1, Lmb2, LmbG2, Lmb3, LmbdG2, LmbG3, LmbDG2, f,
    k1, mu1, ph1,
    k2, mu2, ph2,
    kgrid, Pgrid,
    ):
    k12, mu12, ph12 = addVectors(k1, mu1, ph1, k2, mu2, ph2)
    
    logvk = np.log(np.array([k1, k2, k12]))
    logvPk = linear_interpolate(np.log(kgrid), np.log(Pgrid), logvk)
    vPk = np.exp(logvPk)

    A = 0
    X = 0
    if not np.isclose(k12, 0):
        A = (
            8 * vPk[0]**2 * vZ1(Lmb1, f, k1, mu1, ph1)**2
            * vPk[2] * vZ2(Lmb1, Lmb2, LmbG2, f, k1, -mu1, ph1 + np.pi, k12, mu12, ph12)**2
            + 8 * vPk[1]**2 * vZ1(Lmb1, f, k2, mu2, ph2)**2
            * vPk[2] * vZ2(Lmb1, Lmb2, LmbG2, f, k2, -mu2, ph2 + np.pi, k12, mu12, ph12)**2
        )
        X = (
            16 * vPk[0] * vZ1(Lmb1, f, k1, mu1, ph1)
            * vPk[1] * vZ1(Lmb1, f, k2, mu2, ph2)
            * vPk[2] * vZ2(Lmb1, Lmb2, LmbG2, f, k1, -mu1, ph1 + np.pi, k12, mu12, ph12)
            * vZ2(Lmb1, Lmb2, LmbG2, f, k2, -mu2, ph2 + np.pi, k12, mu12, ph12)
        )

    Star = (
        12 * vZ1(Lmb1, f, k1, mu1, ph1)**2 * vPk[0]**2
        * vZ1(Lmb1, f, k2, mu2, ph2) * vPk[1]
        * vZ3(Lmb1, Lmb2, LmbG2, Lmb3, LmbdG2, LmbG3, LmbDG2, f, k1, mu1, ph1, k1, -mu1, ph1 + np.pi, k2, mu2, ph2)
        + 12 * vZ1(Lmb1, f, k2, mu2, ph2)**2 * vPk[1]**2
        * vZ1(Lmb1, f, k1, mu1, ph1) * vPk[0]
        * vZ3(Lmb1, Lmb2, LmbG2, Lmb3, LmbdG2, LmbG3, LmbDG2, f, k2, mu2, ph2, k1, -mu2, ph2 + np.pi, k1, mu1, ph1)
    )
    return (
        # A
        # + X
        + Star
    )


########################
# Integration routines #
########################


@njit
def integrand_2h_22(
    Dphi,
    k1,
    k2,
    mu1,
    mu2,
    L2mb1,
    f,
    sigma_par,
    sigma_perp,
    k,
    Pgrid,
):
    hTs = 0

    k13, mu13, ph13 = addVectors(k1, mu1, Dphi, k2, mu2, 0.0)
    if not np.isclose(k13, 0):
        Wres = np.exp(
            -2 * k13**2 * (sigma_par**2 * mu13**2 + sigma_perp**2 * (1 - mu13**2))
        )
        P22 = PowerSpectrumLO(L2mb1, L2mb1, f, k13, mu13, ph13, k, Pgrid)
        hTs = 2 * P22 * Wres

    return hTs


@njit
def isotropized_2h_31(
    k1,
    xi,
    Lmb1,
    L3mb1,
    f,
    sigma_par,
    sigma_perp,
    hI1,
    hI3,
    k,
    Pgrid,
):
    lxi = len(xi)
    sigmasq = sigma_par**2 * xi**2 + sigma_perp**2 * (1 - xi**2)
    Wres1 = np.exp(-2 * k1**2 * sigmasq)

    I = np.empty((lxi, lxi))
    for imu1 in range(lxi):
        for imu2 in range(lxi):
            P31 = PowerSpectrumLO(Lmb1, L3mb1, f, k1, xi[imu1], 0.0, k, Pgrid)
            hTs = 2 * P31 * Wres1 * hI3[imu2, imu1] * hI1[imu1]
            I[imu1, imu2] = hTs
    return I


@njit
def integrand_3h(
    Dphi,
    k1,
    k2,
    mu1,
    mu2,
    Lmb1,
    Lmb2,
    LmbG2,
    L2mb1,
    L2mb2,
    L2mbG2,
    f,
    sigma_par,
    sigma_perp,
    k,
    Pgrid,
):

    k13, mu13, ph13 = addVectors(k1, mu1, Dphi, k2, mu2, 0.0)
    Wres = (
        np.exp(-(k1**2) * (sigma_par**2 * mu1**2 + sigma_perp**2 * (1 - mu1**2)))
        * np.exp(-(k2**2) * (sigma_par**2 * mu2**2 + sigma_perp**2 * (1 - mu2**2)))
        * np.exp(-(k13**2) * (sigma_par**2 * mu13**2 + sigma_perp**2 * (1 - mu13**2)))
    )

    hTs = 4 * (
        BispectrumLO(
            Lmb1,
            Lmb2,
            LmbG2,
            L2mb1,
            L2mb2,
            L2mbG2,
            f,
            k1,
            mu1,
            Dphi,
            k2,
            mu2,
            0.0,
            k,
            Pgrid,
        )
        * Wres
    )

    return hTs


@njit
def integrand_4h(
    Dphi,
    k1,
    k2,
    mu1,
    mu2,
    Lmb1,
    Lmb2,
    LmbG2,
    Lmb3,
    LmbdG2,
    LmbG3,
    LmbDG2,
    f,
    sigma_par,
    sigma_perp,
    k,
    Pgrid,
):
    Wres = np.exp(
        -2 * k1**2 * (sigma_par**2 * mu1**2 + sigma_perp**2 * (1 - mu1**2))
    ) * np.exp(
        -2 * k2**2 * (sigma_par**2 * mu2**2 + sigma_perp**2 * (1 - mu2**2))
    )

    hTs = (
        collapsed_Trispectrum_LO(
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
            Dphi,
            k2,
            mu2,
            0.0,
            k,
            Pgrid,
        )
        * Wres
    )
    return hTs


######################
# Integration method #
######################


@njit
def Dphi_integrand(f, xi, args=(), eps=1e-2):
    nxi = len(xi)

    # Compute Delta phi integral
    I = np.empty((nxi, nxi))
    for i in range(nxi):
        mu1 = xi[i]
        for j in range(nxi):
            mu2 = xi[j]
            exArgs = (*args[:2], mu1, mu2, *args[2:])
            # print(*exArgs[:-2])
            I[i, j] = (
                adaptive_mesh_integral(
                    0,
                    np.pi,
                    f,
                    args=exArgs,
                    eps=eps,
                )
                / np.pi
            )
    return I


@njit
def compute_multipole_matrix(xi, wi, I, hI):
    nxi = len(xi)

    ells = [0, 2, 4]  # Hardcoded right now
    # Precompute Legendre polynomials
    L_vals = np.zeros((3, nxi))
    for idx, l in enumerate(ells):
        L_vals[idx, :] = get_legendre(l, xi) * (2 * l + 1) / 2

    pseudo_Cov = np.empty((3, 3))
    for il in range(len(ells)):
        for jl in range(len(ells)):
            L1 = L_vals[il, :, None]
            L2 = L_vals[jl, None, :]
            W = wi[None, :] * wi[:, None]

            pseudo_Cov[il, jl] = np.sum(W * L1 * L2 * I * hI)

    return pseudo_Cov


@njit(parallel=True)
def integrate(f, k, xi, wi, hI, args=(), eps=1e-2):
    kl = len(k)

    isotropized = np.zeros((kl, kl, 3, 3))
    for ik1 in prange(kl):
        for ik2 in prange(ik1, kl):
            I = Dphi_integrand(f, xi, (k[ik1], k[ik2], *args), eps=eps)
            isotropized[ik1, ik2, :, :] = compute_multipole_matrix(
                xi, wi, I, hI[ik1, ik2, :, :]
            )
    isotropized += np.transpose(isotropized, (1, 0, 3, 2))

    return isotropized
