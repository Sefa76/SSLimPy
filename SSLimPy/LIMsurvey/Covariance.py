from functools import partial

import numpy as np
import vegas
from astropy import constants as c
from astropy import units as u
from scipy.integrate import trapezoid
from scipy.special import legendre

import SSLimPy.cosmology.cosmology as cosmo
import SSLimPy.LIMsurvey.PowerSpectra as powspe
from SSLimPy.interface import config as cfg
from SSLimPy.LIMsurvey.higherorder import TrispectrumL0
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
        self.z = powerSpectrum.z
        self.tracer = cfg.settings["TracerPowerSpectrum"]

    def _integrate_4h(self,k1,k2):
        k = self.k
        z = self.z
        Pk = self.cosmo.matpow(k, z, nonlinear="False", tracer=self.tracer)

        # fill starting arguments
        partialTS = partial(TrispectrumL0,
                            k1=k1.to(u.Mpc**-1).value,
                            k2=k1.to(u.Mpc**-1).value,
                            k3=k2.to(u.Mpc**-1).value,
                            k4=k2.to(u.Mpc**-1).value,
                            kgrid=k.to(u.Mpc**-1).value,
                            Pgrid=Pk.to(u.Mpc**3).value,
                            )

        # We only consider squeezed quadrilaterals
        def squeezedTS(x):
            """Returns the 4point halo correlator for a squeezed quadrilateral configuration"""
            mu1 = x[0]
            phi1 = x[1]
            mu2 = x[2]
            phi2 = x[3]
            partialTS(mu1=mu1, ph1=phi1, mu2=-mu1, ph2=phi1 + np.pi,
                      mu3=mu2, ph3=phi2, mu4=-mu2, ph4=phi2 + np.pi)

        muv, phiv =[-1,1] , [-np.pi, np.pi]
        integ = vegas.Integrator([muv, phiv, muv, phiv])
        result = integ(squeezedTS, nitn=10, neval=1000)

        if cfg.settings["verbose"]>2:
            print(result.summary())
            print('result = %s    Q = %.2f' % (result, result.Q))

        I1 = self.powerSpectrum.halomoments(k1, z, bias_order=1, moment=1)
        I2 = self.powerSpectrum.halomoments(k2, z, bias_order=1, moment=1)

        return result * I1**2 * I2**2

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
