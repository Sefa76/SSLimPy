import numpy as np
from SSLimPy.utils.utils import *
from numba import njit, prange

######################################
# (symetrised) mode coupling kernels #
######################################


@njit(
    "(float64,float64,float64,float64,float64)", fastmath=True
)
def _F2(k1, mu1, k2, mu2, Dphi):
    """Unsymetrised F2 kernel"""
    k1pk2 = scalarProduct(k1, mu1, Dphi, k2, mu2, 0.0)
    F2 = (
        5 / 7
        + 1 / 7 * (6 / k1**2 + 1 / k2**2) * k1pk2
        + 2 / 7 * k1pk2**2 / (k1 * k2) ** 2
    )
    return F2


@njit(
    "(float64,float64,float64,float64,float64)", fastmath=True
)
def vF2(k1, mu1, k2, mu2, Dphi):
    """Computes the F2 mode coupling kernel
    All computations are done on a vector grid
    """
    F2 = _F2(k1, mu1, k2, mu2, Dphi)
    F2 += _F2(k2, mu2, k1, mu1, -Dphi)
    F2 *= 1 / 2
    return F2


@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64)",
    fastmath=True,
)
def _F3_T1_symetrised_12(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """in this combination the ir divergence can be resummed
    symetrsed in the first and second argument
    """
    k12, mu12, ph12 = addVectors(k1, mu1, ph1, k2, mu2, ph2)
    if np.isclose(k12, 0):
        return 0
    else:
        k, mu, ph = addVectors(k3, mu3, ph3, k12, mu12, ph12)

        F1 = 1 / 3 * 1 / k3**2
        F2T1 = 1 / 21 * scalarProduct(k1, mu1, ph1, k2, mu2, ph2) / (k1**2 * k2**2)
        F2T2 = (
            1 / 28
            * (
                scalarProduct(k1, mu1, ph1, k12, mu12, ph12) / (k1**2 * k12**2)
                + scalarProduct(k2, mu2, ph2, k12, mu12, ph12) / (k2**2 * k12**2)
            )
        )
        F3 = 7 * k3**2 * scalarProduct(
            k12, mu12, ph12, k, mu, ph
        ) + k**2 * scalarProduct(k3, mu3, ph3, k12, mu12, ph12)
        return F1 * (F2T1 + F2T2) * F3


@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64)",
    fastmath=True,
)
def _F3_T1_symetrised(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """Fully symetrized first term in the F3"""
    F3_T1 = _F3_T1_symetrised_12(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    F3_T1 += _F3_T1_symetrised_12(k3, mu3, ph3, k1, mu1, ph1, k2, mu2, ph2)
    F3_T1 += _F3_T1_symetrised_12(k2, mu2, ph2, k3, mu3, ph3, k1, mu1, ph1)
    F3_T1 *= 1 / 3
    return F3_T1


@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64)",
    fastmath=True,
)
def _F3_T2_symetrised_23(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """in this combination the ir divergence can be resummed
    symetrsed in the second and third argument
    """
    k23, mu23, ph23 = addVectors(k2, mu2, ph2, k3, mu3, ph3)
    if np.isclose(k23, 0):
        return 0
    else:
        k, mu, ph = addVectors(k1, mu1, ph1, k23, mu23, ph23)

        F1 = 1 / 3 * 1 / k1**2 * (k**2 * scalarProduct(k1, mu1, ph1, k23, mu23, ph23))
        F2T1 = 1 / 21 * scalarProduct(k2, mu2, ph2, k3, mu3, ph3) / (k2**2 * k3**2)
        F2T2 = (
            1 / 28
            * (
                scalarProduct(k2, mu2, ph2, k23, mu23, ph23) / (k2**2 * k23**2)
                + scalarProduct(k3, mu3, ph3, k23, mu23, ph23) / (k3**2 * k23**2)
            )
        )
        return F1 * (F2T1 + F2T2)


@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64)",
    fastmath=True,
)
def _F3_T2_symetrised(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """Fully symetrized second term in the F3"""
    F3_T2 = _F3_T2_symetrised_23(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    F3_T2 += _F3_T2_symetrised_23(k3, mu3, ph3, k1, mu1, ph1, k2, mu2, ph2)
    F3_T2 += _F3_T2_symetrised_23(k2, mu2, ph2, k3, mu3, ph3, k1, mu1, ph1)
    F3_T2 *= 1 / 3
    return F3_T2


@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64)",
    fastmath=True,
)
def _F3_T3_symetrised_23(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """Thrid term of the F3 mode coupling kernel
    symetrsed in the second and third argument
    """
    k23, mu23, ph23 = addVectors(k2, mu2, ph2, k3, mu3, ph3)
    if np.isclose(k23, 0):
        return 0
    else:
        k, mu, ph = addVectors(k1, mu1, ph1, k23, mu23, ph23)

        F1 = scalarProduct(k1, mu1, ph1, k, mu, ph) / (18 * k1**2)
        F2T1 = scalarProduct(k2, mu2, ph2, k3, mu3, ph3) * k23**2 / (k2**2 * k3**2)
        F2T2 = (
            5 / 2
            * (
                scalarProduct(k2, mu2, ph2, k23, mu23, ph23) / k2**2
                + scalarProduct(k3, mu3, ph3, k23, mu23, ph23) / k3**2
            )
        )
        return F1 * (F2T1 + F2T2)


@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64)",
    fastmath=True,
)
def _F3_T3_symetrised(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """Fully symetrized third term in the F3"""
    F3_T3 = _F3_T3_symetrised_23(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    F3_T3 += _F3_T3_symetrised_23(k3, mu3, ph3, k1, mu1, ph1, k2, mu2, ph2)
    F3_T3 += _F3_T3_symetrised_23(k2, mu2, ph2, k3, mu3, ph3, k1, mu1, ph1)
    F3_T3 *= 1 / 3
    return F3_T3


@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64)",
    fastmath=True,
)
def vF3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """Computes the F3 mode coupling kernel
    All computations are done on a vector grid
    """
    F3 = _F3_T1_symetrised(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    F3 += _F3_T2_symetrised(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    F3 += _F3_T3_symetrised(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)

    return F3


#######################
# N-point correlators #
#######################


def BispectrumLO(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3, kgrid, Pgrid):
    """Computes the tree level Bispectrum"""
    # Obtain the Power Spectra
    vk = np.array([k1, k2, k3])
    # powerlaw extrapoltion
    vlogP = linear_interpolate(np.log(kgrid), np.log(Pgrid), np.log(vk))
    vP = np.exp(vlogP)

    # Compute over all permutations of F2 diagrams
    Tp1 = vP[0] * vP[1] * vF2(k1, mu1, k2, mu2, ph1 - ph2)
    Tp2 = vP[0] * vP[2] * vF2(k1, mu1, k3, mu3, ph1 - ph3)
    Tp3 = vP[1] * vP[3] * vF2(k2, mu2, k3, mu3, ph2 - ph3)

    return 2 * (Tp1 + Tp2 + Tp3)

@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64[::1], float64[::1])",
    fastmath=True,
)
def TrispectrumL0(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3, k4, mu4, ph4, kgrid, Pgrid):
    """Compute the tree level Trispectrum"""
    # Compute coordinates of added wavevectors
    k12, mu12, ph12 = addVectors(k1, mu1, ph1, k2, mu2, ph2)
    k13, mu13, ph13 = addVectors(k1, mu1, ph1, k3, mu3, ph3)
    k14, mu14, ph14 = addVectors(k1, mu1, ph1, k4, mu4, ph4)
    k23, mu23, ph23 = addVectors(k2, mu2, ph2, k3, mu3, ph3)
    k24, mu24, ph24 = addVectors(k2, mu2, ph2, k4, mu4, ph4)
    k34, mu34, ph34 = addVectors(k3, mu3, ph3, k4, mu4, ph4)

    # Obtain the Power Spectra
    vk = np.array([k1, k2, k3, k4, k12, k13, k14, k23, k24, k34])
    # powerlaw extrapoltion
    vlogP = linear_interpolate(np.log(kgrid), np.log(Pgrid), np.log(vk))
    vP = np.exp(vlogP)

    T1 = 0
    # Compute over all permutations of F2 F2 diagrams
    if not np.isclose(k12, 0):
        T1 += (
            vP[0] * vP[3] * vP[4]
            * vF2(k12, mu12, k1, -mu1, ph12 - ph1 - np.pi)
            * vF2(k34, mu34, k4, -mu4, ph34 - ph4 - np.pi)
        )
        T1 += (
            vP[1] * vP[3] * vP[4]
            * vF2(k12, mu12, k2, -mu2, ph12 - ph2 - np.pi)
            * vF2(k34, mu34, k4, -mu4, ph34 - ph4 - np.pi)
        )
        T1 += (
            vP[2] * vP[1] * vP[9]
            * vF2(k34, mu34, k3, -mu3, ph34 - ph3 - np.pi)
            * vF2(k12, mu12, k2, -mu2, ph12 - ph2 - np.pi)
        )
        T1 += (
            vP[3] * vP[1] * vP[9]
            * vF2(k34, mu34, k4, -mu4, ph34 - ph4 - np.pi)
            * vF2(k12, mu12, k2, -mu2, ph12 - ph2 - np.pi)
        )
    if not np.isclose(k13, 0):
        T1 += (
            vP[0] * vP[1] * vP[5]
            * vF2(k13, mu13, k1, -mu1, ph13 - ph1 - np.pi)
            * vF2(k24, mu24, k2, -mu2, ph24 - ph2 - np.pi)
        )
        T1 += (
            vP[2] * vP[3] * vP[5]
            * vF2(k13, mu13, k3, -mu3, ph13 - ph3 - np.pi)
            * vF2(k24, mu24, k4, -mu4, ph24 - ph4 - np.pi)
        )
        T1 += (
            vP[1] * vP[2] * vP[8]
            * vF2(k24, mu24, k2, -mu2, ph24 - ph2 - np.pi)
            * vF2(k13, mu13, k3, -mu3, ph13 - ph3 - np.pi)
        )
        T1 += (
            vP[3] * vP[0] * vP[8]
            * vF2(k24, mu24, k4, -mu4, ph24 - ph4 - np.pi)
            * vF2(k13, mu13, k1, -mu1, ph13 - ph1 - np.pi)
        )
    if not np.isclose(k14, 0):
        T1 += (
            vP[0] * vP[2] * vP[6]
            * vF2(k14, mu14, k1, -mu1, ph14 - ph1 - np.pi)
            * vF2(k23, mu23, k3, -mu3, ph23 - ph3 - np.pi)
        )
        T1 += (
            vP[3] * vP[2] * vP[6]
            * vF2(k14, mu14, k4, -mu4, ph14 - ph4 - np.pi)
            * vF2(k23, mu23, k3, -mu3, ph23 - ph3 - np.pi)
        )
        T1 += (
            vP[1] * vP[0] * vP[7]
            * vF2(k23, mu23, k2, -mu2, ph23 - ph2 - np.pi)
            * vF2(k14, mu14, k1, -mu1, ph14 - ph1 - np.pi)
        )
        T1 += (
            vP[2] * vP[0] * vP[7]
            * vF2(k23, mu23, k3, -mu3, ph23 - ph3 - np.pi)
            * vF2(k14, mu14, k1, -mu1, ph14 - ph1 - np.pi)
        )
    T1 *= 4
    # That should be all of them ...

    # Compute over all permutations of F3 diagrams
    T2 = vP[0] * vP[1] * vP[2] * vF3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    T2 += vP[1] * vP[2] * vP[3] * vF3(k2, mu2, ph2, k3, mu3, ph3, k4, mu4, ph4)
    T2 += vP[2] * vP[3] * vP[0] * vF3(k3, mu3, ph3, k4, mu4, ph4, k1, mu1, ph1)
    T2 += vP[3] * vP[0] * vP[1] * vF3(k4, mu4, ph4, k1, mu1, ph1, k2, mu2, ph2)
    T2 *= 6

    # print(T1, T2)
    if np.isnan(T1) or np.isnan(T2):
        print(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3, k4, mu4, ph4, kgrid, Pgrid)
        raise RuntimeError("NaN encounterd")

    return T1 + T2


@njit(
    "(float64[::1], float64[::1], float64[::1], float64[::1], float64[::1])",
    parallel=True,
)
def integrate_Trispectrum(k, mu, phi, kgrid, Pgrid):
    kl = len(k)
    mul = len(mu)
    phil = len(phi)

    # obtain neccessary legendre functions
    L0 = legendre_0(mu) * (2 * 0 + 1) / 2
    L2 = legendre_2(mu) * (2 * 2 + 1) / 2
    L4 = legendre_4(mu) * (2 * 4 + 1) / 2

    pseudo_Cov = np.zeros((kl, kl, 3, 3))
    for ik1 in prange(kl):
        for ik2 in prange(ik1, kl):

            mu1_integ0 = np.empty(mul)
            mu1_integ2 = np.empty(mul)
            mu1_integ4 = np.empty(mul)
            for imu1 in range(mul):
                mu2_integ = np.empty(mul)
                for imu2 in range(mul):
                    phi1_integ = np.empty(phil)
                    for iphi1 in range(phil):
                        phi2_integ = np.empty(phil)
                        for iphi2 in range(phil):
                            phi2_integ[iphi2] = TrispectrumL0(
                                k[ik1],
                                mu[imu1],
                                phi[iphi1],
                                k[ik1],
                                -mu[imu1],
                                phi[iphi1] + np.pi,
                                k[ik2],
                                mu[imu2],
                                phi[iphi2],
                                k[ik2],
                                -mu[imu2],
                                phi[iphi2] + np.pi,
                                kgrid,
                                Pgrid,
                            )
                        phi1_integ[iphi1] = gauss_legendre(
                            phi2_integ, phi, -np.pi, np.pi
                        )
                    mu2_integ[imu2] = gauss_legendre(phi1_integ, phi, -np.pi, np.pi)
                # integrate over mu2 first
                mu1_integ0[imu1] = gauss_legendre(mu2_integ * L0, mu, -1, 1)
                mu1_integ2[imu1] = gauss_legendre(mu2_integ * L2, mu, -1, 1)
                mu1_integ4[imu1] = gauss_legendre(mu2_integ * L4, mu, -1, 1)
            pseudo_Cov[ik1, ik2, 0, 0] = gauss_legendre(mu1_integ0 * L0, mu, -1, 1)
            pseudo_Cov[ik1, ik2, 0, 1] = gauss_legendre(mu1_integ2 * L0, mu, -1, 1)
            pseudo_Cov[ik1, ik2, 0, 2] = gauss_legendre(mu1_integ4 * L0, mu, -1, 1)
            pseudo_Cov[ik1, ik2, 1, 0] = gauss_legendre(mu1_integ0 * L2, mu, -1, 1)
            pseudo_Cov[ik1, ik2, 1, 1] = gauss_legendre(mu1_integ2 * L2, mu, -1, 1)
            pseudo_Cov[ik1, ik2, 1, 2] = gauss_legendre(mu1_integ4 * L2, mu, -1, 1)
            pseudo_Cov[ik1, ik2, 2, 0] = gauss_legendre(mu1_integ0 * L4, mu, -1, 1)
            pseudo_Cov[ik1, ik2, 2, 1] = gauss_legendre(mu1_integ2 * L4, mu, -1, 1)
            pseudo_Cov[ik1, ik2, 2, 2] = gauss_legendre(mu1_integ4 * L4, mu, -1, 1)

            # use symetries k1 <-> k2
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
