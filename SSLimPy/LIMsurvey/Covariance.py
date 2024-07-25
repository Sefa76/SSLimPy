import numpy as np
from scipy.integrate import trapezoid
from scipy.special import legendre

import SSLimPy.cosmology.cosmology as cosmo
import SSLimPy.LIMsurvey.PowerSpectra as powspe

from SSLimPy.utils.utils import construct_gaussian_cov

class Covariance:
    def __init__(self, fiducialcosmology: cosmo.cosmo_functions, powerspectrum: powspe.PowerSpectra):
        self.cosmology = fiducialcosmology
        self.powerspectrum = powerspectrum
        self.k = self.powerspectrum.k
        dk = self.powerspectrum.dk
        self.dk = np.append(dk, dk[-1])
        self.mu = self.powerspectrum.mu
        self.z = self.powerspectrum.z

    def Nmodes(self):
        Vk = 4 * np.pi * self.k**2 * self.dk
        _, Vw = self.powerspectrum.Wsurvey(self.k,self.mu)
        return Vk[:, None] * Vw[None, :] / (2 * (2 * np.pi)**3)

    def gaussian_cov(self):
        Pobs = self.powerspectrum.Pk_Obs
        sigma = Pobs**2 / self.Nmodes()[:,None,:]

        # compute the C_ell covaraiance
        cov_00 = trapezoid(
            legendre(0)(self.mu)[None,:,None]**2
            * sigma, axis=1) * (1 / 2)
        cov_20 = trapezoid(
            legendre(0)(self.mu)[None,:,None]
            * legendre(2)(self.mu)[None,:,None]
            * sigma, axis=1) * (5 / 2)
        cov_40 = trapezoid(
            legendre(0)(self.mu)[None,:,None]
            * legendre(4)(self.mu)[None,:,None]
            * sigma, axis=1) * (9 / 2)
        cov_22 = trapezoid(
            legendre(2)(self.mu)[None,:,None]**2
            * sigma, axis=1) * (25 / 2)
        cov_42 = trapezoid(
            legendre(2)(self.mu)[None,:,None]
            * legendre(4)(self.mu)[None,:,None]
            * sigma, axis=1) * (45 / 2)
        cov_44 = trapezoid(
            legendre(4)(self.mu)[None,:,None]**2
            * sigma, axis=1) * (81 / 2)

        # construct the covariance
        nk = np.uint16(len(self.k))
        nz = np.uint16(len(self.z))
        cov = construct_gaussian_cov(nk, nz,
                                     cov_00, cov_20, cov_40,
                                     cov_22, cov_42,
                                     cov_44)
        return cov