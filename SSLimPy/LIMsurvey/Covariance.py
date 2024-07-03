from astropy import units as u
from astropy import constants as c
from scipy.special import j1, spherical_jn
from scipy.interpolate import RectBivariateSpline
from time import time

import numpy as np
from SSLimPy.interface import config as cfg
from copy import copy


class Covariance:
    def __init__(self, fiducialcosmology, powerspectrum):
        self.cosmology = fiducialcosmology
        self.powerspectrum = powerspectrum

    def Lfield(self, z1, z2):
        zgrid = [z1, z2]
        Lgrid = self.cosmology.comoving(zgrid)
        return Lgrid[1] - Lgrid[0]

    def Sfield(self, zc, Omegafield):
        r2 = np.power(self.cosmology.comoving(zc).to(u.Mpc), 2)
        sO = Omegafield.to(u.rad**2).value
        return r2 * sO

    def Wsurvey(self, q, muq):
        """Compute the Fourier-transformed sky selection window function"""
        q = np.atleast_1d(q)
        muq = np.atleast_1d(muq)

        qparr = q[:, None, None] * muq[None, :, None]
        qperp = q[:, None, None] * np.sqrt(1 - np.power(muq[None, :, None], 2))

        # Calculate dz from deltanu
        nu = self.powerspectrum.nu
        nuObs = self.powerspectrum.nuObs
        Delta_nu = cfg.obspars["Delta_nu"]
        z = (nu / nuObs - 1).to(1).value
        z_min = (nu / (nuObs + Delta_nu / 2) - 1).to(1).value
        z_max = (nu / (nuObs - Delta_nu / 2) - 1).to(1).value

        # Construct W_survey (now just a cylinder)
        Sfield = self.Sfield(z, cfg.obspars["Omega_field"])
        Lperp = np.sqrt( Sfield/ np.pi)
        Lparr = self.Lfield(z_min, z_max)
        Wperp = 2 * np.pi * Lperp * j1((qperp * Lperp).to(1).value) / qperp
        Wparr = Lparr * spherical_jn(0, (qparr * Lparr / 2).to(1).value)

        Wsurvey = Wperp * Wparr
        Vsurvey = Sfield * Lperp
        return Wsurvey, Vsurvey

    def convolved_Pk(self):
        """Convolves the Observed power spectrum with the survey volume

        This enters as the gaussian term in the final covariance
        """
        # Extract the pre-computed power spectrum
        k = self.powerspectrum.k
        mu = self.powerspectrum.mu
        Pobs = self.powerspectrum.Pk_Obs

        # Downsample q, muq and deltaphi
        q = np.geomspace(
            k[0], k[-1],
            np.uint8(len(k) / cfg.settings["downsample_conv_q"])
        )
        muq = np.linspace(
            -1, 1,
            np.uint8((len(mu) + 1) / cfg.settings["downsample_conv_muq"])
        )
        muq = (muq[1:] + muq[:-1]) / 2.0
        deltaphi = np.linspace(-np.pi, np.pi, 2 * len(muq))

        # Obtain survey Window
        Wsurvey, Vsurvey = self.Wsurvey(q, muq)

        Pconv = np.zeros_like(Pobs)
        # Do the convolution for each redshift bin
        for iz in range(Pobs.shape[-1]):
            # Obtain a Interpolator of the Observed PS for these redshift to compute the P(k-q)
            logk = np.log(k.to(u.Mpc**-1).value)
            logP = np.log(Pobs.value)
            Pofz = RectBivariateSpline(logk, mu, logP[Ellipsis, iz], kx=1, ky=1)

            # Do the 3D integration over q (for now) in order phi, mu, q
            q_integrant = np.zeros((*Pobs[Ellipsis, iz].shape, *q.shape))
            for iq, qi in enumerate(q):
                mu_integrant = np.zeros((*Pobs[Ellipsis, iz].shape, *muq.shape))
                for imuq, muqi in enumerate(muq):
                    abskminusq = np.sqrt(
                        k[:, None, None] ** 2
                        + qi**2
                        - 2
                        * (
                            qi * muqi * k[:, None, None] * mu[None, :, None]
                            + qi
                            * np.sqrt(1 - muqi**2)
                            * k[:, None, None]
                            * np.sqrt(1 - mu[None, :, None] ** 2)
                            * np.cos(deltaphi)[None, None, :]
                        )
                    )
                    mukminusq = (
                        qi * muqi + k[:, None, None] * mu[None, :, None]
                    ) / abskminusq
                    logPofphi = Pofz(
                        np.log(abskminusq.to(u.Mpc**-1).value),
                        mukminusq,
                        grid=False,
                    )
                    phiintegrant = 1 / (2*np.pi)**3 * qi**2 * np.abs(Wsurvey[iq, imuq, iz])**2 * np.exp(logPofphi)*Pobs.unit
                    mu_integrant[: , :, imuq] = np.trapz(phiintegrant, deltaphi, axis=-1)
                q_integrant[:, :, iq] = np.trapz(mu_integrant, muq, axis=-1)
            Pconv[:, :, iz] = np.trapz(q_integrant * q[None, None, :].to(u.Mpc**-1), np.log(q.to(u.Mpc**-1).value), axis=-1)
        print("done!")
        return Pconv / Vsurvey
