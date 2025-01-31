import itertools
from numba import njit, prange
import numpy as np
import SSLimPy.LIMsurvey.PowerSpectra as pobs
from astropy import constants as c
from astropy import units as u
from scipy.integrate import trapezoid
from scipy.special import legendre, roots_legendre
from SSLimPy.interface import config as cfg
from SSLimPy.LIMsurvey.higherorder import _integrate_2h, _integrate_3h, _integrate_4h
from SSLimPy.utils.utils import *


class Covariance:
    def __init__(self, powerspectrum: pobs.PowerSpectra):
        self.cosmology = powerspectrum.fiducialcosmo
        self.powerspectrum = powerspectrum
        self.k = self.powerspectrum.k
        dk = self.powerspectrum.dk
        self.dk = np.append(dk, dk[-1])
        self.mu = self.powerspectrum.mu
        self.z = self.powerspectrum.z

    def Nmodes(self):
        Vk = 4 * np.pi * self.k**2 * self.dk
        _, Vw = self.powerspectrum.Wsurvey(self.k, self.mu)
        return Vk[:, None] * Vw[None, :] / (2 * (2 * np.pi) ** 3)

    def Detector_noise(self):
        F1 = (
            cfg.obspars["Tsys_NEFD"] ** 2
            * cfg.obspars["Omega_field"].to(u.sr).value
            / (2 * cfg.obspars["nD"] * cfg.obspars["tobs"])
        )
        F2 = c.c / cfg.obspars["nu"]
        F3 = (
            self.cosmology.comoving(self.z) ** 2
            * (1 + self.z) ** 2
            / self.cosmology.Hubble(self.z, physical=True)
        )
        PI = (F1 * F2 * F3).to(self.powerspectrum.Pk_Obs.unit)
        return PI

    def gaussian_cov(self):
        Pobs = self.powerspectrum.Pk_Obs
        PI = self.Detector_noise()
        sigma = (Pobs + PI) ** 2 / self.Nmodes()[:, None, :]

        # compute the C_ell covaraiance
        cov_00 = trapezoid(legendre(0)(self.mu)[None, :, None] ** 2 * sigma, axis=1) * (
            1 / 2
        )
        cov_20 = trapezoid(
            legendre(0)(self.mu)[None, :, None]
            * legendre(2)(self.mu)[None, :, None]
            * sigma,
            axis=1,
        ) * (5 / 2)
        cov_40 = trapezoid(
            legendre(0)(self.mu)[None, :, None]
            * legendre(4)(self.mu)[None, :, None]
            * sigma,
            axis=1,
        ) * (9 / 2)
        cov_22 = trapezoid(legendre(2)(self.mu)[None, :, None] ** 2 * sigma, axis=1) * (
            25 / 2
        )
        cov_42 = trapezoid(
            legendre(2)(self.mu)[None, :, None]
            * legendre(4)(self.mu)[None, :, None]
            * sigma,
            axis=1,
        ) * (45 / 2)
        cov_44 = trapezoid(legendre(4)(self.mu)[None, :, None] ** 2 * sigma, axis=1) * (
            81 / 2
        )

        # construct the covariance
        nk = np.uint16(len(self.k))
        nz = np.uint16(len(self.z))
        cov = construct_gaussian_cov(
            nk, nz, cov_00, cov_20, cov_40, cov_22, cov_42, cov_44
        )
        return cov * (Pobs**2).unit


class nonGuassianCov:
    def __init__(self, powerSpectrum: pobs.PowerSpectra):
        self.cosmo = powerSpectrum.cosmology
        self.astro = powerSpectrum.astro
        self.powerSpectrum = powerSpectrum
        self.k = powerSpectrum.k
        self.mu = powerSpectrum.mu
        self.z = powerSpectrum.z
        self.tracer = cfg.settings["TracerPowerSpectrum"]

    def integrate_4h(self):
        k = self.k
        z = self.z
        Pk = self.cosmo.matpow(k, z, nonlinear="False", tracer=self.tracer)

        xi, w = roots_legendre(cfg.settings["nnodes_legendre"])
        mu = np.pi * xi

        kl = len(k)
        wl = len(w)

        # compute I1 for v_of_M models
        I1 = np.empty((kl, wl))
        indexmenge = range(wl)
        for imu1 in indexmenge:
            Ii = self.powerSpectrum.reduced_halo_temperature_moments(
                z, k, mu[imu1], moment=1
            )
            I1[:, imu1] = Ii.value

        Lmb1 = self.astro.bavghalo("b1", z, 1)
        Lmb2 = self.astro.bavghalo("b2", z, 1)
        LmbG2 = self.astro.bavghalo("bG2", z, 1)
        Lmb3 = self.astro.bavghalo("b3", z, 1)
        LmbdG2 = self.astro.bavghalo("bdG2", z, 1)
        LmbG3 = self.astro.bavghalo("bG3", z, 1)
        LmbDG2 = self.astro.bavghalo("bDG2", z, 1)
        f = self.cosmo.growth_rate(1e-3 * u.Mpc**-1, z, tracer=self.tracer)

        k, Pk = k.value, Pk.value

        result = _integrate_4h(
            Lmb1, Lmb2, LmbG2, Lmb3, LmbdG2, LmbG3, LmbDG2, f, xi, w, k, Pk, I1
        )

        return result

    def integrate_3h(self):
        k = self.k
        z = self.z
        Pk = self.cosmo.matpow(k, z, nonlinear="False", tracer=self.tracer)

        xi, w = roots_legendre(cfg.settings["nnodes_legendre"])
        mu = xi * np.pi

        #########################
        # Precompute Halo terms #
        #########################
        kl = len(k)
        wl = len(w)

        # compute I1 for v_of_M models (We cant compute fNL models)
        indexmenge = range(wl)
        I1 = np.empty((kl, wl))
        for imu1 in indexmenge:
            Ii = self.powerSpectrum.halo_temperature_moments(
                z, k, mu[imu1], bias_order=1, moment=1
            )
            I1[:, imu1] = Ii.value

        indexmenge = itertools.product(range(wl), repeat=2)
        I2 = np.empty((kl, kl, wl, wl))
        for (
            imu1,
            imu2,
        ) in indexmenge:
            Iij = self.powerSpectrum.halo_temperature_moments(
                z, k, k, mu[imu1], mu[imu2], bias_order=1, moment=2
            )
            I2[:, :, imu1, imu2] = Iij.value

        k, Pk = k.value, Pk.value

        result = _integrate_3h(k, xi, w, Pk, I1, I2)

        return result

    def integrate_2h(self):
        k = self.k
        z = self.z
        Pk = self.cosmo.matpow(k, z, nonlinear="False", tracer=self.tracer)

        xi, w = roots_legendre(cfg.settings["nnodes_legendre"])
        mu = xi * np.pi

        #########################
        # Precompute Halo terms #
        #########################
        kl = len(k)
        wl = len(w)

        # compute I1 for v_of_M models (We cant compute fNL models)
        indexmenge = range(wl)
        I1 = np.empty((kl, wl))
        for imu1 in indexmenge:
            Ii = self.powerSpectrum.halo_temperature_moments(
                z, k, mu[imu1], bias_order=1, moment=1
            )
            I1[:, imu1] = Ii.value

        indexmenge = itertools.product(range(wl), repeat=2)
        I2 = np.empty((kl, kl, wl, wl))
        for (
            imu1,
            imu2,
        ) in indexmenge:
            Iij = self.powerSpectrum.halo_temperature_moments(
                z, k, k, mu[imu1], mu[imu2], bias_order=1, moment=2
            )
            I2[:, :, imu1, imu2] = Iij.value

        indexmenge = itertools.product(range(wl), repeat=2)
        I3 = np.empty((kl, kl, wl, wl))
        for (
            imu1,
            imu2,
        ) in indexmenge:
            # the third argument is allways the negative of the first
            for ik, ki in enumerate(k):
                Iijk = self.powerSpectrum.halo_temperature_moments(
                    z, ki, k, ki, mu[imu1], mu[imu2], -mu[imu1], bias_order=1, moment=3
                )
                I3[ik, :, imu1, imu2] = Iijk.value

        k, Pk = k.value, Pk.value

        result = _integrate_2h(k, xi, w, Pk, I1, I2, I3)

        return result

    def integrate_1h(self):
        k = self.k
        z = self.z
        Pk = self.cosmo.matpow(k, z, nonlinear="False", tracer=self.tracer)

        xi, w = roots_legendre(cfg.settings["nnodes_legendre"])
        mu = xi * np.pi

        #########################
        # Precompute Halo terms #
        #########################
        kl = len(k)
        wl = len(w)

        indexmengemu = itertools.product(range(wl), repeat=2)
        indexmengek = itertools.product(range(kl), repeat=2)
        I4 = np.empty((kl, kl, wl, wl))
        for (
            imu1,
            imu2,
        ) in indexmengemu:
            for ik1, ik2 in indexmengek:
                Iij = self.powerSpectrum.halo_temperature_moments(
                    z,
                    k[ik1],
                    k[ik2],
                    k[ik1],
                    k[ik2],
                    mu[imu1],
                    mu[imu2],
                    -mu[imu1],
                    -mu[imu2],
                    bias_order=1,
                    moment=4,
                )
                I4[ik1, ik2, imu1, imu2] = Iij.value

        result = np.sum(np.sum(I4 / 4 * w, axis=-1) * w, axis=-1)
        return result


##############
# Numba part #
##############


@njit(
    "(uint16, uint16, "
    + "float64[:,:], float64[:,:], float64[:,:], "
    + "float64[:,:], float64[:,:], "
    + "float64[:,:])",
    parallel=True,
)
def construct_gaussian_cov(nk, nz, C00, C20, C40, C22, C42, C44):
    cov = np.empty((nk, 3, 3, nz))
    for ki in prange(nk):
        for zi in range(nz):
            cov[ki, 0, 0, zi] = C00[ki, zi]
            cov[ki, 1, 0, zi] = C20[ki, zi]
            cov[ki, 2, 0, zi] = C40[ki, zi]
            cov[ki, 0, 1, zi] = C20[ki, zi]
            cov[ki, 0, 2, zi] = C40[ki, zi]
            cov[ki, 1, 1, zi] = C22[ki, zi]
            cov[ki, 1, 2, zi] = C42[ki, zi]
            cov[ki, 2, 1, zi] = C42[ki, zi]
            cov[ki, 2, 2, zi] = C44[ki, zi]
    return cov
