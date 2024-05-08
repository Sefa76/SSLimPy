"""
Calculate the halo mass function as function of mass for different fitting functions

All functions return dn/dM(M)

Each function takes: self, Mvec, rhoM

Takes inspiration from pylians
"""

import numpy as np
from scipy.interpolate import interp1d


def ST(self, Mvec, rhoM):
    """
    Sheth-Tormen halo mass function
    """
    deltac = 1.686
    nu = (deltac / self.sigmaM) ** 2.0
    nup = 0.707 * nu

    dndM = (
        -2
        * (rhoM / Mvec)
        * self.dsigmaM_dM.to(self.Msunh**-1)
        / self.sigmaM
        * 0.3222
        * (1.0 + 1.0 / nup**0.3)
        * np.sqrt(0.5 * nup)
        * np.exp(-0.5 * nup)
        / np.sqrt(np.pi)
    )

    return dndM


def Tinker(self, Mvec, rhoM):
    """
    Tinker et al 2008 halo mass function for delta=200
    """
    delta = 200
    alpha = 10 ** (-((0.75 / np.log10(delta / 75.0)) ** 1.2))

    # this is for R200_critical with OmegaM=0.2708
    D = 200.0
    A = 0.186 * (1.0 + self.z) ** (-0.14)
    a = 1.47 * (1.0 + self.z) ** (-0.06)
    b = 2.57 * (1.0 + self.z) ** (-alpha)
    c = 1.19

    fs = A * ((b / self.sigmaM) ** (a) + 1.0) * np.exp(-c / self.sigmaM**2)

    dndM = -(rhoM / Mvec) * self.dsigmaM_dM.to(self.Msunh**-1) * fs / self.sigmaM

    return dndM


def Crocce(self, Mvec, rhoM):
    """
    Crocce et al. halo mass function
    """
    A = 0.58 * (1.0 + self.z) ** (-0.13)
    a = 1.37 * (1.0 + self.z) ** (-0.15)
    b = 0.3 * (1.0 + self.z) ** (-0.084)
    c = 1.036 * (1.0 + self.z) ** (-0.024)

    fs = A * (self.sigmaM ** (-a) + b) * np.exp(-c / self.sigmaM**2)
    dndM = -(rhoM / Mvec) * self.dsigmaM_dM.to(self.Msunh**-1) * fs / self.sigmaM

    return dndM


def Jenkins(self, Mvec, rhoM):
    """
    Jenkins et al. halo mass function
    """
    A = 0.315
    b = 0.61
    c = 3.8

    fs = A * np.exp(-np.absolute(np.log(1.0 / self.sigmaM) + b) ** c)

    dndM = -(rhoM / Mvec) * self.dsigmaM_dM.to(self.Msunh**-1) * fs / self.sigmaM

    return dndM


def Warren(self, Mvec, rhoM):
    """
    Warren et al. halo mass function
    """
    A = 0.7234
    a = 1.625
    b = 0.2538
    c = 1.1982

    fs = A * (self.sigmaM ** (-a) + b) * np.exp(-c / self.sigmaM**2)
    dndM = -(rhoM / Mvec) * self.dsigmaM_dM.to(self.Msunh**-1) * fs / self.sigmaM

    return dndM


def Watson(self, Mvec, rhoM):
    """
    Watson et al. halo mass function for delta=200. Can be changed
    """
    delta = 200.0
    OmegaM = (
        rhoM
        / 2.77536627e11
        * (self.Msunh * self.Mpch**-3).to(self.Msunh * self.Mpch**-3)
    )

    A = 0.194
    a = 1.805
    b = 2.267
    c = 1.287

    factor = (
        np.exp(0.023 * (delta / 178.0 - 1.0))
        * (delta / 178.0) ** (-0.456 * OmegaM - 0.139)
        * np.exp(0.072 * (1 - delta / 178.0) / self.sigmaM**2.130)
    )
    fs = A * (self.sigmaM ** (-a) + b) * np.exp(-c / self.sigmaM**2) * factor

    dndM = -(rhoM / Mvec) * self.dsigmaM_dM.to(self.Msunh**-1) * fs / self.sigmaM

    return dndM


def Watson_FOF(self, Mvec, rhoM):
    """
    Watson et al. halo mass function using FOF
    """
    A = 0.282
    a = 2.163
    b = 1.406
    c = 1.210

    fs = A * ((b / self.sigmaM) ** a + 1.0) * np.exp(-c / self.sigmaM**2)

    dndM = -(rhoM / Mvec) * self.dsigmaM_dM.to(self.Msunh**-1) * fs / self.sigmaM

    return dndM


def Angulo(self, Mvec, rhoM):
    """
    Angulo et al. halo mass function
    """
    fs = 0.265 * (1.675 / self.sigmaM + 1.0) ** 1.9 * np.exp(-1.4 / self.sigmaM**2)

    dndM = -(rhoM / Mvec) * self.dsigmaM_dM.to(self.Msunh**-1) * fs / self.sigmaM

    return dndM
