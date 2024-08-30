from numba import njit, prange
import numpy as np

import SSLimPy.cosmology.cosmology as cosmology
import SSLimPy.cosmology.astro as astrophysics

from SSLimPy.utils.utils import _linear_interpolate, _trapezoid
from SSLimPy.interface import config as cfg


class nonGuassianCov:
    def __init__(
        self, cosmo: cosmology.cosmo_functions, astro: astrophysics.astro_functions
    ):
        self.cosmo = cosmo
        self.astro = astro
        self.tracer = cfg.settings["TracerPowerSpectrum"]

def addVectors(k1, mu1, ph1,
               k2, mu2, ph2,
               ):
    k1pk2 = (
        k1 * k2 * (np.sqrt((1 - mu1**2) * (1 - mu2**2)) * np.cos(ph1 - ph2) + mu1 * mu2)
    )
    k12 = np.sqrt(k1**2 + 2 * k1pk2 + k2**2)
    mu12 = (k1 * mu1 + k2 * mu2) / k12
    phi12 = np.arctan2(k1 * np.sqrt(1 - mu1**2) * np.sin(ph1) + k2 * np.sqrt(1 - mu2**2) * np.sin(ph2),
                       k1 * np.sqrt(1 - mu1**2) * np.cos(ph1) + k2 * np.sqrt(1 - mu2**2) * np.cos(ph2))
    return k12, mu12, phi12

def vF2(k1, mu1, k2, mu2, Dphi):
    """Computes the F2 mode coupling kernel
    All computations are done on a vector grid
    """
    k1pk2 = k1 * k2 * (np.sqrt((1 - mu1**2) * (1 - mu2**2)) * np.cos(Dphi) + mu1 * mu2)

    F2 = (
        5 / 7
        + 1 / 2 * (1 / k1**2 + 1 / k2**2) * k1pk2
        + 2 / 7 * k1pk2**2 / (k1 * k2) ** 2
    )
    return F2

def vG2(k1, mu1, k2, mu2, Dphi):
    """Computes the G2 mode coupling kernel
    All computations are done on a vector grid
    """
    k1pk2 = k1 * k2 * (np.sqrt((1 - mu1**2) * (1 - mu2**2)) * np.cos(Dphi) + mu1 * mu2)

    G2 = (
        3 / 5
        + 1 / 2 * (1 / k1**2 + 1 / k2**2) * k1pk2
        + 4 / 7 * k1pk2**2 / (k1 * k2) ** 2
    )
    return G2

def vF3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3):
    """Computes the F3 mode coupling kernel
    All computations are done on a vector grid
    """
    k1pk2 = (
        k1 * k2 * (np.sqrt((1 - mu1**2) * (1 - mu2**2)) * np.cos(ph1 - ph2) + mu1 * mu2)
    )

    T1 = (
        7
        / 18
        * (k1**2 + k1pk2)
        / k1**2
        * (vF2(k2, mu2, k3, mu3, ph2 - ph3) + vG2(k1, mu1, k2, mu2, ph1 - ph2))
    )
    T2 = 1 / 18 * (k1**2 + 2 * k1pk2 + k2**2) * k1pk2 / (k1 * k2) ** 2 + (
        vG2(k2, mu2, k3, mu3, ph2 - ph3) + vG2(k1, mu1, k2, mu2, ph1 - ph2)
    )
    return T1 + T2

def BispectrumLO(k1, mu1, ph1,
                 k2, mu2, ph2,
                 k3, mu3, ph3,
                 kgrid, Pgrid):
    """Computes the tree level Bispectrum
    """
    # Obtain the Power Spectra
    vk = np.array([k1, k2, k3])
    vP = _linear_interpolate(kgrid, Pgrid, vk)

    # Compute over all permutations of F2 diagrams
    Tp1 = vP[0] * vP[1] * vF2(k1, mu1, k2, mu2, ph1 - ph2)
    Tp2 = vP[0] * vP[2] * vF2(k1, mu1, k3, mu3, ph1 - ph3)
    Tp3 = vP[1] * vP[3] * vF2(k2, mu2, k3, mu3, ph2 - ph3)

    return Tp1 + Tp2 + Tp3

def TrispectrumL0(k1, mu1, ph1,
                  k2, mu2, ph2,
                  k3, mu3, ph3,
                  k4, mu4, ph4,
                  kgrid, Pgrid):
    """ Compute the tree level Trispectrum
    """
    # Compute coordinates of added wavevectors 
    k12, mu12, ph12 = addVectors(k1, mu1, ph1, k2, mu2, ph2)
    k13, mu13, ph13 = addVectors(k1, mu1, ph1, k3, mu3, ph3)
    k14, mu14, ph14 = addVectors(k1, mu1, ph1, k4, mu4, ph4)
    k23, mu23, ph23 = addVectors(k2, mu2, ph2, k3, mu3, ph3)
    k24, mu24, ph24 = addVectors(k2, mu2, ph2, k4, mu4, ph4)
    k34, mu34, ph34 = addVectors(k3, mu3, ph3, k4, mu4, ph4)

    # Obtain the Power Spectra
    vk = np.array([k1, k2, k3, k4,
                   k12, k13, k14,
                   k23, k24,
                   k34])
    vP = _linear_interpolate(kgrid, Pgrid, vk)

    # Compute over all ṕermutations of F2 F2 diagrams
    T1 = vP[0] * vP[3] * vP[4] * vF2(k12, mu12, k1, -mu1, ph12 - ph1 - np.pi) * vF2(k34, mu34, k4, -mu4, ph34 - ph4 - np.pi)
    T1 += vP[0] * vP[1] * vP[5] * vF2(k13, mu13, k1, -mu1, ph13 - ph1 - np.pi) * vF2(k24, mu24, k2, -mu2, ph24 - ph2 - np.pi)
    T1 += vP[0] * vP[2] * vP[6] * vF2(k14, mu14, k1, -mu1, ph14 - ph1 - np.pi) * vF2(k23, mu23, k3, -mu3, ph23 - ph3 - np.pi)
    T1 += vP[1] * vP[0] * vP[7] * vF2(k23, mu23, k2, -mu2, ph23 - ph2 - np.pi) * vF2(k14, mu14, k1, -mu1, ph14 - ph1 - np.pi)
    T1 += vP[1] * vP[2] * vP[8] * vF2(k24, mu24, k2, -mu2, ph24 - ph2 - np.pi) * vF2(k13, mu13, k3, -mu3, ph13 - ph3 - np.pi)
    T1 += vP[1] * vP[3] * vP[4] * vF2(k12, mu12, k2, -mu2, ph12 - ph2 - np.pi) * vF2(k34, mu34, k4, -mu4, ph34 - ph4 - np.pi)
    T1 += vP[2] * vP[1] * vP[9] * vF2(k34, mu34, k3, -mu3, ph34 - ph3 - np.pi) * vF2(k12, mu12, k2, -mu2, ph12 - ph2 - np.pi)
    T1 += vP[2] * vP[3] * vP[5] * vF2(k13, mu13, k3, -mu3, ph13 - ph3 - np.pi) * vF2(k24, mu24, k4, -mu4, ph24 - ph4 - np.pi)
    T1 += vP[2] * vP[0] * vP[7] * vF2(k23, mu23, k3, -mu3, ph23 - ph3 - np.pi) * vF2(k14, mu14, k1, -mu1, ph14 - ph1 - np.pi)
    T1 += vP[3] * vP[2] * vP[6] * vF2(k14, mu14, k4, -mu4, ph14 - ph4 - np.pi) * vF2(k23, mu23, k3, -mu3, ph23 - ph3 - np.pi)
    T1 += vP[3] * vP[0] * vP[8] * vF2(k24, mu24, k4, -mu4, ph24 - ph4 - np.pi) * vF2(k13, mu13, k1, -mu1, ph13 - ph1 - np.pi)
    T1 += vP[3] * vP[1] * vP[9] * vF2(k34, mu34, k4, -mu4, ph34 - ph4 - np.pi) * vF2(k12, mu12, k2, -mu2, ph12 - ph2 - np.pi)
    T1 *= 4
    # That should be all of them ...

    # Compute over all permutations of F3 diagrams
    T2 = vP[0] * vP[1] * vP[2] * vF3(k1, mu1, ph1, k2, mu2, ph2, k3, mu3, ph3)
    T2 += vP[1] * vP[2] * vP[3] * vF3(k2, mu2, ph2, k3, mu3, ph3, k4, mu4, ph4)
    T2 += vP[2] * vP[3] * vP[0] * vF3(k3, mu3, ph3, k4, mu4, ph4, k1, mu1, ph1)
    T2 += vP[3] * vP[0] * vP[1] * vF3(k4, mu4, ph4, k1, mu1, ph1, k2, mu2, ph2)
    T2 *= 6

    return T1 + T2