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
        "(float64, float64, float64,"+
        "float64, float64, float64)",
        fastmath=True,
        )
def Galileon2(k1, mu1, ph1, k2, mu2, ph2):
    """Second Galileon in Fourier space
    """
    muC12 = scalarProduct(k1,mu1, ph1, k2, mu2, ph2) / (k1 * k2)
    return muC12**2 - 1

@njit(
        "(float64, float64, float64,"+
        "float64, float64, float64,"+
        "float64, float64, float64)",
        fastmath=True,
        )
def Galileon3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """Third Galileon in Fourier space
    """
    muC12 = scalarProduct(k1,mu1, ph1, k2, mu2, ph2) / (k1 * k2)
    muC23 = scalarProduct(k2,mu2, ph2, k3, mu3, ph3) / (k2 * k3)
    muC13 = scalarProduct(k1,mu1, ph1, k3, mu3, ph3) / (k1 * k3)
    return 1 + - muC12**2 - muC23**2 - muC13**2 + 2 * muC12 * muC23 * muC13

######################################
# (symetrised) mode coupling kernels #
######################################


@njit("(float64,float64,float64,float64,float64,float64)", fastmath=True)
def vF2(k1, mu1, ph1, k2, mu2, ph2):
    """Symetrised F2 kernel"""
    k1pk2 = scalarProduct(k1, mu1, ph1, k2, mu2, ph2)
    F2 = (
        5 / 7
        + 1 / 2 * (1 / k1**2 + 1 / k2**2) * k1pk2
        + 2 / 7 * k1pk2**2 / (k1 * k2) ** 2
    )
    return F2


@njit("(float64,float64,float64,float64,float64,float64)", fastmath=True)
def vG2(k1, mu1, ph1, k2, mu2, ph2):
    """Symetrised G2 kernel"""
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
    """in this combination the ir divergence can be resummed
    symetrsed in the second and third argument
    """
    k23, mu23, ph23 = addVectors(k2, mu2, ph2, k3, mu3, ph3)
    if np.isclose(k23, 0):
        return 0
    else:
        F2_23 = vF2(k2, mu2, ph2, k3, mu3, ph3)
        G2_23 = vG2(k2, mu2, ph2, k3, mu3, ph3)
        a23 = alpha(k1, mu1, ph1, k23, mu23, ph23)
        b23 = beta(k1, mu1, ph1, k23, mu23, ph23)

        result = (
                1 / 18 * (
                    7 * a23 * F2_23 + 2 * b23 * G2_23
                    )
                )
        return result


@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64)",
    fastmath=True,
)
def _F3_T2_symetrised_12(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """in this combination the ir divergence can be resummed
    symetrsed in the first and second argument
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
    """Computes the F3 mode coupling kernel
    """
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
    """in this combination the ir divergence can be resummed
    symetrsed in the second and third argument
    """
    k23, mu23, ph23 = addVectors(k2, mu2, ph2, k3, mu3, ph3)
    if np.isclose(k23, 0):
        return 0
    else:
        F2_23 = vF2(k2, mu2, ph2, k3, mu3, ph3)
        G2_23 = vG2(k2, mu2, ph2, k3, mu3, ph3)
        a23 = alpha(k1, mu1, ph1, k23, mu23, ph23)
        b23 = beta(k1, mu1, ph1, k23, mu23, ph23)

        result = (
                1 / 18 * (
                    3 * a23 * F2_23 + 6 * b23 * G2_23
                    )
                )
        return result


@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64)",
    fastmath=True,
)
def _G3_T2_symetrised_12(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """in this combination the ir divergence can be resummed
    symetrsed in the first and second argument
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
    """Computes the G3 mode coupling kernel
    """
    G3 = _G3_T1_symetrised_23(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    G3 += _G3_T1_symetrised_23(k2, mu2, ph2, k3, mu3, ph3, k1, mu1, ph1)
    G3 += _G3_T1_symetrised_23(k3, mu3, ph3, k1, mu1, ph1, k2, mu2, ph2)
    G3 += _G3_T2_symetrised_12(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    G3 += _G3_T2_symetrised_12(k2, mu2, ph2, k3, mu3, ph3, k1, mu1, ph1)
    G3 += _G3_T2_symetrised_12(k3, mu3, ph3, k1, mu1, ph1, k2, mu2, ph2)
    return G3 / 3


###############
# RSD Kernals #
###############
# These functions are for the appoximation that the mean bias is roughly independent
# from the k-depedent shape inside these integrals

@njit("(float64, float64,"
      +"float64, float64, float64)",
      fastmath=True,
      )
def vZ1(mb1, f, k1, mu1, ph1):
    """Kaiser term RSD function
    """
    return mb1 + f * mu1**2


@njit("(float64, float64, float64, float64,"
      +"float64, float64, float64,"
      +"float64, float64, float64)",
      fastmath=True,
      )
def vZ2(mb1, mb2, mbG2, f, k1, mu1, ph1, k2, mu2, ph2):
    """Second order RSD mode coupling kernal
    """
    k12, mu12, ph12 = addVectors(k1, mu1, ph1, k2, mu2, ph2)
    z2 = mb1 * vF2(k1, mu1, ph1, k2, mu2, ph2)
    z2 += mb2 / 2
    z2 += mbG2 * Galileon2(k1, mu1, ph1, k2, mu2, ph2)
    z2 += f * mu12**2 * vG2(k1, mu1, ph1, k2, mu2, ph2)
    z2 += f * mu12 * k12 / 2 * mb1 * (mu1 / k1 + mu2 / k2)
    z2 += (f * mu12 * k12)**2 / 2 * mu1 / k1 * mu2 / k2

    return z2


@njit("(float64, float64, float64,"
      +"float64, float64, float64, float64,"
      +"float64,"
      +"float64, float64, float64,"
      +"float64, float64, float64,"
      +"float64, float64, float64)",
      fastmath=True,
      )
def _Z3_symetrised_23(mb1, mb2, mb3, mbG2, mbdG2, mbG3, mbDG2, f, k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """ Thrid order RSD mode coupling kernel symetrized in the second and third argument
    possible divergences in k23 are captured by the mode coupling kernels in this way
    """
    k12, mu12, ph12 = addVectors(k1, mu1, ph1, k2, mu2, ph2)
    k23, mu23, ph23 = addVectors(k2, mu2, ph2, k3, mu3, ph3)
    k, mu, ph = addVectors(k1, mu1, ph1, k23, mu23, ph23)

    z3 = mb1 * vF3(k1, mu1, ph1, k2, mu2, ph2)
    z3 += mb2 * vF2(k2, mu2, ph2, k3, mu3, ph3)
    z3 += mb3 / 6
    z3 += mbdG2 * Galileon2(k2, mu2, ph2, k3, mu3, ph3)
    z3 += mbG3 * Galileon3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    z3 += f * mu**2 * vG3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    z3 += f * k * mu * mu1 / k1 * (
            mb1 * vF2(k2, mu2, ph2, k3, mu3, ph3) 
            + mb2 / 2 
            + mbG2 * Galileon2(k2, mu2, ph2, k3, mu3, ph3)
            )
    z3 += (f * k * mu)**2 / 2 * mb1 * mu2 / k2 * mu3 / k3
    z3 += (f * k * mu)**6 / 6 * mu1 / k1 * mu2 / k2 * mu3 / k3
    if not np.isclose(k23,0):
        z3 += 2 * mbG2 * Galileon2(k1, mu1, ph1, k23, mu23, ph23) * vF2(k2, mu2, ph2, k3, mu3, ph3)
        z3 += f * k * mu * mb1 * mu23 / k23 * vG2(k2, mu2, ph2, k3, mu3, ph3)
        z3 += mbDG2 * Galileon2(k1, mu1, ph1, k23, mu23, ph23) * (
                vF2(k2, mu2, ph2, k3, mu3, ph3) - vG2(k2, mu2, ph2, k3, mu3, ph3)
                )
        z3 += (f * k * mu)**2 * mu1 / k1 * mu23/ k23 * vG2(k2, mu2, ph2, k3, mu3, ph3)
    return z3

@njit("(float64, float64, float64,"
      +"float64, float64, float64, float64,"
      +"float64,"
      +"float64, float64, float64,"
      +"float64, float64, float64,"
      +"float64, float64, float64)",
      fastmath=True,
      )
def vZ3(mb1, mb2, mb3, mbG2, mbdG2, mbG3, mbDG2, f, k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    Z3  = _Z3_symetrised_23(mb1, mb2, mb3, mbG2, mbdG2, mbG3, mbDG2, f, k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    Z3 += _Z3_symetrised_23(mb1, mb2, mb3, mbG2, mbdG2, mbG3, mbDG2, f, k2, mu2, ph2, k3, mu3, ph3, k1, mu1, ph1)
    Z3 += _Z3_symetrised_23(mb1, mb2, mb3, mbG2, mbdG2, mbG3, mbDG2, f, k3, mu3, ph3, k1, mu1, ph1, k2, mu2, ph2)
    Z3 *= 1 / 3
    return Z3

#######################
# N-point correlators #
#######################

@njit(
    "(float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64,float64,float64,"
    + "float64[::1], float64[::1])",
    fastmath=True,
)
def BispectrumLO(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3, kgrid, Pgrid):
    """Computes the tree level Bispectrum"""
    # Obtain the Power Spectra
    vk = np.array([k1, k2, k3])
    # powerlaw extrapoltion
    vlogP = linear_interpolate(np.log(kgrid), np.log(Pgrid), np.log(vk))
    vP = np.exp(vlogP)
    # Compute over all permutations of F2 diagrams
    T = 0
    if not np.isclose(k1, 0) and not np.isclose(k2, 0):
        v1 = vF2(k1, mu1, ph1, k2, mu2, ph2)
        T += vP[0] * vP[1] * v1
    if not np.isclose(k1, 0) and not np.isclose(k3, 0):
        v2 = vF2(k1, mu1,  ph1, k3, mu3, ph3)
        T += vP[0] * vP[2] * v2
    if not np.isclose(k2, 0) and not np.isclose(k3, 0):
        v3 = vF2(k2, mu2, ph2, k3, mu3, ph3)
        T += vP[1] * vP[2] * v3

    return 2 * T


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
            vP[0]
            * vP[3]
            * vP[4]
            * vF2(k12, mu12, ph12 , k1, -mu1, ph1 + np.pi)
            * vF2(k34, mu34, ph34 , k4, -mu4, ph4 + np.pi)
        )
        T1 += (
            vP[1]
            * vP[3]
            * vP[4]
            * vF2(k12, mu12, ph12, k2, -mu2, ph2 + np.pi)
            * vF2(k34, mu34, ph34, k4, -mu4, ph4 + np.pi)
        )
        T1 += (
            vP[2]
            * vP[1]
            * vP[9]
            * vF2(k34, mu34, ph34, k3, -mu3, ph3 + np.pi)
            * vF2(k12, mu12, ph12, k2, -mu2, ph2 + np.pi)
        )
        T1 += (
            vP[3]
            * vP[1]
            * vP[9]
            * vF2(k34, mu34, ph34, k4, -mu4, ph4 + np.pi)
            * vF2(k12, mu12, ph12, k2, -mu2, ph2 + np.pi)
        )
    if not np.isclose(k13, 0):
        T1 += (
            vP[0]
            * vP[1]
            * vP[5]
            * vF2(k13, mu13, ph13, k1, -mu1, ph1 + np.pi)
            * vF2(k24, mu24, ph24, k2, -mu2, ph2 + np.pi)
        )
        T1 += (
            vP[2]
            * vP[3]
            * vP[5]
            * vF2(k13, mu13, ph13, k3, -mu3, ph3 + np.pi)
            * vF2(k24, mu24, ph24, k4, -mu4, ph4 + np.pi)
        )
        T1 += (
            vP[1]
            * vP[2]
            * vP[8]
            * vF2(k24, mu24, ph24, k2, -mu2, ph2 + np.pi)
            * vF2(k13, mu13, ph13, k3, -mu3, ph3 + np.pi)
        )
        T1 += (
            vP[3]
            * vP[0]
            * vP[8]
            * vF2(k24, mu24, ph24, k4, -mu4, ph4 + np.pi)
            * vF2(k13, mu13, ph13, k1, -mu1, ph1 + np.pi)
        )
    if not np.isclose(k14, 0):
        T1 += (
            vP[0]
            * vP[2]
            * vP[6]
            * vF2(k14, mu14, ph14, k1, -mu1, ph1 + np.pi)
            * vF2(k23, mu23, ph23, k3, -mu3, ph3 + np.pi)
        )
        T1 += (
            vP[3]
            * vP[2]
            * vP[6]
            * vF2(k14, mu14, ph14, k4, -mu4, ph4 + np.pi)
            * vF2(k23, mu23, ph23, k3, -mu3, ph3 + np.pi)
        )
        T1 += (
            vP[1]
            * vP[0]
            * vP[7]
            * vF2(k23, mu23, ph23, k2, -mu2, ph2 + np.pi)
            * vF2(k14, mu14, ph14, k1, -mu1, ph1 + np.pi)
        )
        T1 += (
            vP[2]
            * vP[0]
            * vP[7]
            * vF2(k23, mu23, ph23, k3, -mu3,  ph3 + np.pi)
            * vF2(k14, mu14, ph14, k1, -mu1,  ph1 + np.pi)
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



########################
# Integration routines #
########################

@njit(
        "(float64[::1], float64[::1], float64[::1], float64[::1], float64[:,:], float64[:,:,:,:], float64[:,:,:,:])"
)
def _integrate_2h(k, xi, w, Pgrid, I1grid, I2grid, I3grid):
    assert len(xi) == len(w), "Number of integration points must match number of weights"
    nnodes = len(xi)
    kl = len(k)

    mu = xi
    phi = np.pi * xi

    # obtain neccessary legendre functions
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
                            # 22 halos

                            # Obtain the Power Spectra
                            k13, _, _ = addVectors(k[ik1], mu[imu1], phi[iphi1], k[ik2], mu[imu2], phi[iphi2])
                            k14, _, _ = addVectors(k[ik1], mu[imu1], phi[iphi1], k[ik2], -mu[imu2], phi[iphi2]+np.pi)
                            vk = np.array([k13,k14])
                            # powerlaw extrapoltion
                            vlogP = linear_interpolate(np.log(k), np.log(Pgrid), np.log(vk))
                            vP = np.exp(vlogP)

                            if np.isclose(k13,0):
                                vP[0] = 0
                            if np.isclose(k14,0):
                                vP[1] = 0

                            hT1 = I2grid[ik1,ik2,imu1,imu2] * I2grid[ik1,ik2,imu1,imu2] * (vP[0] + vP[1])

                            # 31 halos
                            hT2 = (
                                2 * I3grid[ik1, ik2, imu1, imu2] * I1grid[ik2, imu2] * Pgrid[ik2]
                                + 2* I3grid[ik2, ik1, imu2, imu1] * I1grid[ik1, imu1] * Pgrid[ik1]
                            )
                            phi2_integ[iphi2] = (hT1 + hT2)/(4 * np.pi) ** 2
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


@njit(
        "(float64[::1], float64[::1], float64[::1], float64[::1], float64[:,:], float64[:,:,:,:])",
    parallel=True,
)
def _integrate_3h(k, xi, w, Pgrid, I1grid, I2grid):
    assert len(xi) == len(w), "Number of integration points must match number of weights"
    nnodes = len(xi)
    kl = len(k)

    mu = xi
    phi = np.pi * xi

    # obtain neccessary legendre functions
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
                            # 1 2
                            # All terms vanish

                            # 1 3
                            kex, muex, phex = addVectors(
                                    k[ik1], mu[imu1], phi[iphi1],
                                    k[ik2], mu[imu2], phi[iphi2]
                                    )
                            B = BispectrumLO(
                                    kex, muex, phex,
                                    k[ik1], -mu[imu1], phi[iphi1] + np.pi,
                                    k[ik2], -mu[imu2], phi[iphi2] + np.pi,
                                    k, Pgrid)
                            I1_1 = I1grid[ik1, imu1] # Symetric in mu -> -mu
                            I1_2 = I1grid[ik2, imu2]
                            I2_3 = I2grid[ik1, ik2, imu1, imu2]
                            phi2_integ[iphi2] = B * I1_1 * I1_2 * I2_3

                            # 1 4
                            kex, muex, phex = addVectors(
                                    k[ik1], mu[imu1], phi[iphi1],
                                    k[ik2], -mu[imu2], phi[iphi2] + np.pi
                                    )
                            B = BispectrumLO(
                                    kex, muex, phex,
                                    k[ik1], -mu[imu1], phi[iphi1] + np.pi,
                                    k[ik2], mu[imu2], phi[iphi2],
                                    k, Pgrid)
                            phi2_integ[iphi2] += B * I1_1 * I1_2 * I2_3

                            # 2 3
                            kex, muex, phex = addVectors(
                                    k[ik1], -mu[imu1], phi[iphi1] + np.pi,
                                    k[ik2], mu[imu2], phi[iphi2]
                                    )
                            B = BispectrumLO(
                                    kex, muex, phex,
                                    k[ik1], mu[imu1], phi[iphi1],
                                    k[ik2], -mu[imu2], phi[iphi2] + np.pi,
                                    k, Pgrid)
                            phi2_integ[iphi2] += B * I1_1 * I1_2 * I2_3

                            # 2 4
                            kex, muex, phex = addVectors(
                                    k[ik1], -mu[imu1], phi[iphi1] + np.pi,
                                    k[ik2], -mu[imu2], phi[iphi2] + np.pi
                                    )
                            B = BispectrumLO(
                                    kex, muex, phex,
                                    k[ik1], mu[imu1], phi[iphi1],
                                    k[ik2], mu[imu2], phi[iphi2],
                                    k, Pgrid)
                            phi2_integ[iphi2] += B * I1_1 * I1_2 * I2_3

                            # 3 4
                            # All terms vanish

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


@njit(
    "(float64[::1], float64[::1], float64[::1], float64[::1], float64[:,:])",
    parallel=True,
)
def _integrate_4h(k, xi, w, Pgrid, I1grid):
    assert len(xi) == len(w), "Number of integration points must match number of weights"
    nnodes = len(xi)
    kl = len(k)

    mu = xi
    phi = np.pi * xi

    # obtain neccessary legendre functions
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
                                    k,
                                    Pgrid,
                                )
                                / (4 * np.pi) ** 2
                                * I1grid[ik1, imu1] ** 2 # Symetric in mu -> -mu
                                * I1grid[ik2, imu2] ** 2
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
