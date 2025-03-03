import itertools
from numba import njit, prange
import numpy as np
from SSLimPy.LIMsurvey import power_spectrum
from astropy import units as u
from scipy.integrate import trapezoid
from scipy.special import legendre, roots_legendre
from SSLimPy.interface import config as cfg
from SSLimPy.LIMsurvey.higher_order import _integrate_2h, _integrate_3h, _integrate_4h
from SSLimPy.utils.utils import *


class Covariance:
    def __init__(self, power_spectrum: power_spectrum.PowerSpectra):
        self.cosmology = power_spectrum.fiducial_cosmology
        self.survey_specs = power_spectrum.survey_specs
        self.power_spectrum = power_spectrum
        self.k = self.power_spectrum.k
        dk = self.power_spectrum.dk
        self.dk = np.append(dk, dk[-1])
        self.mu = self.power_spectrum.mu
        self.z = self.power_spectrum.z

    def Nmodes(self):
        Vk = 4 * np.pi * self.k**2 * self.dk
        _, Vw = self.survey_specs.Wsurvey(self.k, self.mu)
        return Vk[:, None] * Vw[None, :] / (2 * (2 * np.pi) ** 3)

    def Detector_noise(self):
        F1 = (
            self.survey_specs.obsparams["Tsys_NEFD"] ** 2
            * self.survey_specs.obsparams["Omega_field"].to(u.sr).value
            / (2 * self.survey_specs.obsparams["nD"] * self.survey_specs.obsparams["tobs"])
        )
        F2 = self.cosmology.CELERITAS / self.survey_specs.obsparams["nu"]
        F3 = (
            self.cosmology.comoving(self.z) ** 2
            * (1 + self.z) ** 2
            / self.cosmology.Hubble(self.z, physical=True)
        )
        PI = (F1 * F2 * F3).to(self.power_spectrum.Pk_Obs.unit)
        return PI

    def gaussian_cov(self):
        Pobs = self.power_spectrum.Pk_Obs
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
    def __init__(self, power_spectrum: power_spectrum.PowerSpectra):
        self.cosmo = power_spectrum.cosmology
        self.astro = power_spectrum.astro
        self.powerSpectrum = power_spectrum
        self.k = power_spectrum.k
        self.mu = power_spectrum.mu
        self.z = power_spectrum.z
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
            Ii = self.astro.Thalo(
                z, k, mu[imu1], p=1
            )
            I1[:, imu1] = Ii.value

        Lmb1 = self.astro.bavg("b1", z, 1)
        Lmb2 = self.astro.bavg("b2", z, 1)
        LmbG2 = self.astro.bavg("bG2", z, 1)
        Lmb3 = self.astro.bavg("b3", z, 1)
        LmbdG2 = self.astro.bavg("bdG2", z, 1)
        LmbG3 = self.astro.bavg("bG3", z, 1)
        LmbDG2 = self.astro.bavg("bDG2", z, 1)
        f = self.cosmo.growth_rate(1e-3 * u.Mpc**-1, z, tracer=self.tracer)

        k, Pk = k.value, Pk.value

        result = _integrate_4h(
            Lmb1, Lmb2, LmbG2, Lmb3, LmbdG2, LmbG3, LmbDG2, f.item(), xi, w, k, Pk, I1
        )

        return result

    def integrate_3h(self):
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
            Ii = self.astro.Thalo(
                z, k, mu[imu1], p=1
            )
            I1[:, imu1] = Ii.value

        indexmenge = itertools.product(range(wl), repeat=2)
        I2 = np.empty((kl, kl, wl, wl))
        for (
            imu1,
            imu2,
        ) in indexmenge:
            Iij = self.astro.Thalo(
                z, k, k, mu[imu1], mu[imu2], p=2
            )
            I2[:, :, imu1, imu2] = Iij.value

        Lmb1 = self.astro.bavg("b1", z, 1)
        Lmb2 = self.astro.bavg("b2", z, 1)
        LmbG2 = self.astro.bavg("bG2", z, 1)
        L2mb1 = self.astro.bavg("b1", z, 2)
        L2mb2 = self.astro.bavg("b2", z, 2)
        L2mbG2 = self.astro.bavg("bG2", z, 2)
        f = self.cosmo.growth_rate(1e-3 * u.Mpc**-1, z, tracer=self.tracer)

        k, Pk = k.value, Pk.value

        result = _integrate_3h(Lmb1, Lmb2, LmbG2, L2mb1, L2mb2, L2mbG2, f.item(), xi, w, k, Pk, I1, I2)

        return result

    def integrate_2h(self):
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
            Ii = self.astro.Thalo(
                z, k, mu[imu1], p=1
            )
            I1[:, imu1] = Ii.value

        indexmenge = itertools.product(range(wl), repeat=2)
        I2 = np.empty((kl, kl, wl, wl))
        I3 = np.empty((kl, kl, wl, wl))

        for (
            imu1,
            imu2,
        ) in indexmenge:
            Iij = self.astro.Thalo(
                z, k, k, mu[imu1], mu[imu2], p=2
            )
            I2[:, :, imu1, imu2] = Iij.value

            # the third argument is allways the negative of the first
            for ik, ki in enumerate(k):
                Iijk = self.astro.Thalo(
                    z, ki, k, ki, mu[imu1], mu[imu2], -mu[imu1], p=3
                )
                I3[ik, :, imu1, imu2] = Iijk.value

        Lmb1 = self.astro.bavg("b1", z, 1)
        L2mb1 = self.astro.bavg("b1", z, 2)
        L3mb1 = self.astro.bavg("b1", z, 3)
        f = self.cosmo.growth_rate(1e-3 * u.Mpc**-1, z, tracer=self.tracer)

        k, Pk = k.value, Pk.value

        result = _integrate_2h(Lmb1, L2mb1, L3mb1, f.item(), xi, w,k, Pk, I1, I2, I3)

        return result

    def integrate_1h(self):
        k = self.k
        z = self.z

        xi, w = roots_legendre(cfg.settings["nnodes_legendre"])
        mu = np.pi * xi

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
                Iij = self.astro.Thalo(
                    z,
                    k[ik1],
                    k[ik2],
                    k[ik1],
                    k[ik2],
                    mu[imu1],
                    mu[imu2],
                    -mu[imu1],
                    -mu[imu2],
                    p=4,
                )
                I4[ik1, ik2, imu1, imu2] = Iij.value

        result = np.empty((kl, kl, 3, 3))
        result[:,:, 0, 0] = np.sum(np.pi * w * legendre_0(mu) * np.sum(np.pi * w * legendre_0(mu) * I4))
        result[:,:, 0, 1] = np.sum(np.pi * w * legendre_2(mu) * np.sum(np.pi * w * legendre_0(mu) * I4))
        result[:,:, 0, 2] = np.sum(np.pi * w * legendre_4(mu) * np.sum(np.pi * w * legendre_0(mu) * I4))
        result[:,:, 1, 0] = result[:,:, 0, 1]
        result[:,:, 1, 1] = np.sum(np.pi * w * legendre_2(mu) * np.sum(np.pi * w * legendre_2(mu) * I4))
        result[:,:, 1, 2] = np.sum(np.pi * w * legendre_4(mu) * np.sum(np.pi * w * legendre_2(mu) * I4))
        result[:,:, 2, 0] = result[:,:, 0, 2]
        result[:,:, 2, 1] = result[:,:, 1, 2]
        result[:,:, 2, 2] = np.sum(np.pi * w * legendre_4(mu) * np.sum(np.pi * w * legendre_4(mu) * I4))
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
