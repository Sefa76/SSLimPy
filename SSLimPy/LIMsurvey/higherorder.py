import numpy as np

from SSLimPy.utils.utils import _addVectors, _linear_interpolate, _scalarProduct

######################################
# (symetrised) mode coupling kernels #
######################################


def _F2(k1, mu1, k2, mu2, Dphi):
    """Unsymetrised F2 kernel"""
    k1pk2 = _scalarProduct(k1, mu1, Dphi, k2, mu2, 0.0)
    F2 = (
        5 / 7
        + 1 / 7 * (6 / k1**2 + 1 / k2**2) * k1pk2
        + 2 / 7 * k1pk2**2 / (k1 * k2) ** 2
    )
    return F2


def vF2(k1, mu1, k2, mu2, Dphi):
    """Computes the F2 mode coupling kernel
    All computations are done on a vector grid
    """
    F2 = _F2(k1, mu1, k2, mu2, Dphi)
    F2 += _F2(k2, mu2, k1, mu1, -Dphi)
    F2 *= 1 / 2
    return F2

def _F3_T1_symetrised_12(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """in this combination the ir divergence can be resummed
    symetrsed in the first and second argument
    """
    k12, mu12, ph12 = _addVectors(k1, mu1, ph1, k2, mu2, ph2)
    if np.isclose(k12, 0):
        return 0
    else:
        k, mu, ph = _addVectors(k3, mu3, ph3, k12, mu12, ph12)

        F1 = 1 / 3 * 1 / k3**2
        F2T1 = 1 / 21 * _scalarProduct(k1, mu1, ph1, k2, mu2, ph2) / (k1**2 * k2**2)
        F2T2 = (
            1 / 28
            * (
                _scalarProduct(k1, mu1, ph1, k12, mu12, ph12) / (k1**2 * k12**2)
                + _scalarProduct(k2, mu2, ph2, k12, mu12, ph12) / (k2**2 * k12**2)
            )
        )
        F3 = 7 * k3**2 * _scalarProduct(
            k12, mu12, ph12, k, mu, ph
        ) + k**2 * _scalarProduct(k3, mu3, ph3, k12, mu12, ph12)
        return F1 * (F2T1 + F2T2) * F3


def _F3_T1_symetrised(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """Fully symetrized first term in the F3"""
    F3_T1 = _F3_T1_symetrised_12(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    F3_T1 += _F3_T1_symetrised_12(k3, mu3, ph3, k1, mu1, ph1, k2, mu2, ph2)
    F3_T1 += _F3_T1_symetrised_12(k2, mu2, ph2, k3, mu3, ph3, k1, mu1, ph1)
    F3_T1 *= 1 / 3
    return F3_T1


def _F3_T2_symetrised_23(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """in this combination the ir divergence can be resummed
    symetrsed in the second and third argument
    """
    k23, mu23, ph23 = _addVectors(k2, mu2, ph2, k3, mu3, ph3)
    if np.isclose(k23, 0):
        return 0
    else:
        k, mu, ph = _addVectors(k1, mu1, ph1, k23, mu23, ph23)

        F1 = 1 / 3 * 1 / k1**2 * (k**2 * _scalarProduct(k1, mu1, ph1, k23, mu23, ph23))
        F2T1 = 1 / 21 * _scalarProduct(k2, mu2, ph2, k3, mu3, ph3) / (k2**2 * k3**2)
        F2T2 = (
            1 / 28
            * (
                _scalarProduct(k2, mu2, ph2, k23, mu23, ph23) / (k2**2 * k23**2)
                + _scalarProduct(k3, mu3, ph3, k23, mu23, ph23) / (k3**2 * k23**2)
            )
        )
        return F1 * (F2T1 + F2T2)


def _F3_T2_symetrised(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """Fully symetrized second term in the F3"""
    F3_T2 = _F3_T2_symetrised_23(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    F3_T2 += _F3_T2_symetrised_23(k3, mu3, ph3, k1, mu1, ph1, k2, mu2, ph2)
    F3_T2 += _F3_T2_symetrised_23(k2, mu2, ph2, k3, mu3, ph3, k1, mu1, ph1)
    F3_T2 *= 1 / 3
    return F3_T2


def _F3_T3_symetrised_23(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """Thrid term of the F3 mode coupling kernel
    symetrsed in the second and third argument
    """
    k23, mu23, ph23 = _addVectors(k2, mu2, ph2, k3, mu3, ph3)
    if np.isclose(k23, 0):
        return 0
    else:
        k, mu, ph = _addVectors(k1, mu1, ph1, k23, mu23, ph23)

        F1 = _scalarProduct(k1, mu1, ph1, k, mu, ph) / (18 * k1**2)
        F2T1 = _scalarProduct(k2, mu2, ph2, k3, mu3, ph3) * k23**2 / (k2**2 * k3**2)
        F2T2 = (
            5 / 2
            * (
                _scalarProduct(k2, mu2, ph2, k23, mu23, ph23) / k2**2
                + _scalarProduct(k3, mu3, ph3, k23, mu23, ph23) / k3**2
            )
        )
        return F1 * (F2T1 + F2T2)


def _F3_T3_symetrised(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """Fully symetrized third term in the F3"""
    F3_T3 = _F3_T3_symetrised_23(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    F3_T3 += _F3_T3_symetrised_23(k3, mu3, ph3, k1, mu1, ph1, k2, mu2, ph2)
    F3_T3 += _F3_T3_symetrised_23(k2, mu2, ph2, k3, mu3, ph3, k1, mu1, ph1)
    F3_T3 *= 1 / 3
    return F3_T3


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
    vlogP = _linear_interpolate(np.log(kgrid), np.log(Pgrid), np.log(vk))
    vP = np.exp(vlogP)

    # Compute over all permutations of F2 diagrams
    Tp1 = vP[0] * vP[1] * vF2(k1, mu1, k2, mu2, ph1 - ph2)
    Tp2 = vP[0] * vP[2] * vF2(k1, mu1, k3, mu3, ph1 - ph3)
    Tp3 = vP[1] * vP[3] * vF2(k2, mu2, k3, mu3, ph2 - ph3)

    return 2 * (Tp1 + Tp2 + Tp3)


def TrispectrumL0(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3, k4, mu4, ph4, kgrid, Pgrid):
    """Compute the tree level Trispectrum"""
    # Compute coordinates of added wavevectors
    k12, mu12, ph12 = _addVectors(k1, mu1, ph1, k2, mu2, ph2)
    k13, mu13, ph13 = _addVectors(k1, mu1, ph1, k3, mu3, ph3)
    k14, mu14, ph14 = _addVectors(k1, mu1, ph1, k4, mu4, ph4)
    k23, mu23, ph23 = _addVectors(k2, mu2, ph2, k3, mu3, ph3)
    k24, mu24, ph24 = _addVectors(k2, mu2, ph2, k4, mu4, ph4)
    k34, mu34, ph34 = _addVectors(k3, mu3, ph3, k4, mu4, ph4)

    # Obtain the Power Spectra
    vk = np.array([k1, k2, k3, k4, k12, k13, k14, k23, k24, k34])
    # powerlaw extrapoltion
    vlogP = _linear_interpolate(np.log(kgrid), np.log(Pgrid), np.log(vk))
    vP = np.exp(vlogP)

    T1 = 0
    # Compute over all permutations of F2 F2 diagrams
    print(k12,k13, k14)
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
