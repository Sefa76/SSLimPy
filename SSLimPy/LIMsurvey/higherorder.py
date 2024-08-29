from numba import njit, prange
import numpy as np

import SSLimPy.cosmology.cosmology as cosmology
import SSLimPy.cosmology.astro as astrophysics

from SSLimPy.utils.utils import _bilinear_interpolate, _trapezoid
from SSLimPy.interface import config as cfg


class nonGuassianCov:
    def __init__(
        self, cosmo: cosmology.cosmo_functions, astro: astrophysics.astro_functions
    ):
        self.cosmo = cosmo
        self.astro = astro
        self.tracer = cfg.settings["TracerPowerSpectrum"]


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
