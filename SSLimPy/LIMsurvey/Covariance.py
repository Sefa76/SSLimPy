from functools import partial

import numpy as np
import vegas
from astropy import constants as c
from astropy import units as u
from scipy.integrate import trapezoid
from scipy.special import legendre, roots_legendre

import SSLimPy.cosmology.cosmology as cosmo
import SSLimPy.LIMsurvey.PowerSpectra as powspe
from SSLimPy.interface import config as cfg
from SSLimPy.LIMsurvey.higherorder import integrate_Trispectrum
from SSLimPy.utils.utils import construct_gaussian_cov


class Covariance:
    def __init__(self, powerspectrum: powspe.PowerSpectra):
        self.cosmology = powerspectrum.fiducialcosmo
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

    def Detector_noise(self):
        F1 = (
            cfg.obspars["Tsys_NEFD"]**2
            * cfg.obspars["Omega_field"].to(u.sr).value
            / (2 * cfg.obspars["nD"]
               * cfg.obspars["tobs"]
            )
        )
        F2 = c.c / cfg.obspars["nu"]
        F3 = (
            self.cosmology.comoving(self.z)**2
            * (1+ self.z)**2
            / self.cosmology.Hubble(self.z, physical=True)
        )
        PI = (F1 * F2 * F3).to(self.powerspectrum.Pk_Obs.unit)
        return PI

    def gaussian_cov(self):
        Pobs = self.powerspectrum.Pk_Obs
        PI = self.Detector_noise()
        sigma = (Pobs+PI)**2 / self.Nmodes()[:,None,:]

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
        return cov * (Pobs**2).unit

class nonGuassianCov:
    def __init__(
        self, powerSpectrum: powspe.PowerSpectra
    ):
        self.cosmo = powerSpectrum.cosmology
        self.astro = powerSpectrum.astro
        self.powerSpectrum = powerSpectrum
        self.k = powerSpectrum.k
        self.mu = powerSpectrum.mu
        self.z = powerSpectrum.z
        self.tracer = cfg.settings["TracerPowerSpectrum"]

    def _integrate_4h(self):
        k = self.k
        z = self.z
        Pk = self.cosmo.matpow(k, z, nonlinear="False", tracer=self.tracer)

        # Downsample q, muq, and phiq
        nq = np.uint8(len(k) / cfg.settings["downsample_conv_q"])
        if "log" in cfg.settings["k_kind"]:
            q = np.geomspace(k[0], k[-1], nq)
        else:
            q = np.linspace(k[0], k[-1], nq)

        xi, w = roots_legendre(cfg.settings["nnodes_legendre"])
        result = integrate_Trispectrum(q, xi, w, k, Pk) * Pk.unit**2
        
        Il = [self.powerSpectrum.halomoments(ki, z, bias_order=1, moment=1) for ki in k]
        I = np.array([Ii.value for Ii in Il]) * Il[0].unit

        return result * I[:, None, None, None] * I[None, :, None, None]

    def _integrate_3h(self,k1,k2):
        k = self.k
        z = self.z
        Pk = self.cosmo.matpow(k, z, nonlinear="False", tracer=self.tracer)
        pass

    def _integrate_PPT(self,k1,k2):
        k = self.k
        z = self.z
        Pk = self.cosmo.matpow(k, z, nonlinear="False", tracer=self.tracer)
        pass
