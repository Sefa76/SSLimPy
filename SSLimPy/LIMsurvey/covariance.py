import itertools

import numpy as np
from astropy import units as u
from numba import njit, prange
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from scipy.special import legendre, roots_legendre

from SSLimPy.interface import config as cfg
from SSLimPy.LIMsurvey import power_spectrum
from SSLimPy.LIMsurvey.higher_order import *
from SSLimPy.LIMsurvey import ingredients_T0
from SSLimPy.utils.fft_log import FFTLog
from SSLimPy.utils.utils import *


class Covariance:
    def __init__(self, power_spectrum: power_spectrum.PowerSpectra):
        self.cosmology = power_spectrum.fiducial_cosmology
        self.survey_specs = power_spectrum.survey_specs
        self.power_spectrum = power_spectrum
        self.k = self.power_spectrum.k
        self.dk = self.power_spectrum.dk
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
            / (
                2
                * self.survey_specs.obsparams["nD"]
                * self.survey_specs.obsparams["tobs"]
            )
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
        self.tracer = cfg.settings["TracerPowerSpectrum"]
        self.k = power_spectrum.k
        self.mu = power_spectrum.mu
        self.z = power_spectrum.z

        # Get power spectra on grids for numerical computations
        # TODO: For now only works for scale-independent growth
        self.kgrid = power_spectrum.k_numerics.to(u.Mpc**-1)
        self.Pgrid = self.cosmo.matpow(self.kgrid, 0.0, nonlinear=False, tracer=self.tracer).to(u.Mpc**3)
        self.Pk = self.cosmo.matpow(self.k, 0.0, nonlinear=False, tracer=self.tracer).to(u.Mpc**3)

        # FFTlog Approximation
        kminebs = np.min(self.kgrid).to(u.Mpc**-1).value / cfg.settings["Log-extrap"]
        kmaxebs = np.max(self.kgrid).to(u.Mpc**-1).value * cfg.settings["Log-extrap"]

        def extrap_pk(k, kgrid, Pkgrid):
            logk = np.log(k)
            logkgrid = np.log(kgrid)
            logPkgrid = np.log(Pkgrid)
            logPk = linear_interpolate(logkgrid, logPkgrid, logk)
            return np.exp(logPk)

        def p_rec(k, q):
            tf = FFTLog(extrap_pk, kminebs, kmaxebs,
                                    cfg.settings("LogN_modes"), q,
                                    kgrid=self.kgrid.value, Pkgrid=self.Pgrid.value,
                                    )
            return tf(k).real

        #find good numerical bias
        q, _= curve_fit(p_rec, self.k.value, self.P.value, p0=[0.3], sigma=self.Pk.value)
        q = q[0]
        self.fftLog_Pofk = FFTLog(extrap_pk, kminebs, kmaxebs,
                                  cfg.settings("LogN_modes"), q,
                                  kgrid=self.kgrid.value, Pkgrid=self.Pgrid.value,
                                  )


    def integrate_4h(self, args: dict = dict(), z=None):
        k = self.k
        if z is None:
            z = self.z
        z = np.atleast_1d(z)
        D = np.atleast_1d(self.cosmo.growth_factor(1e-4*u.Mpc**-1, z, tracer=self.tracer))

        I1 = restore_shape(self.astro.Thalo(z, k, p=1).value, k, z)
        #This is the normalisation found in pyCCL, Not necessarily true for LIM
        I1 = I1 / I1.value[0, :]

        b1_L1 = np.atleast_1d(args.get("b1", self.astro.bavg("b1", z, 1)))
        b2_L1 = np.atleast_1d(args.get("b2", self.astro.bavg("b2", z, 1)))
        bG2_L1 = np.atleast_1d(args.get("bG2", self.astro.bavg("bG2", z, 1)))
        b3_L1 = np.atleast_1d(args.get("b3", self.astro.bavg("b3", z, 1)))
        bdG2_L1 = np.atleast_1d(args.get("bdG2", self.astro.bavg("bdG2", z, 1)))
        bG3_L1 = np.atleast_1d(args.get("bG3", self.astro.bavg("bG3", z, 1)))
        bDG2_L1 = np.atleast_1d(args.get("bDG2", self.astro.bavg("bDG2", z, 1)))

        kl = len(k)
        zl = len(z)

        # 3111 Terms
        kernel_4h_3111 = np.empty((kl, kl, zl))
        for iz, zi in enumerate(z):
            kernel_4h_3111_iz = ingredients_T0.T3111_kernel(k[:,None], k[None, :], b1_L1[iz], b2_L1[iz], bG2_L1[iz], b3_L1[iz], bdG2_L1[iz], bG3_L1[iz], bDG2_L1[iz])
            squeezed_4h_3111_iz = ingredients_T0.T3111_squeezed(b1_L1[iz], b2_L1[iz], bG2_L1[iz], b3_L1[iz], bdG2_L1[iz], bG3_L1[iz], bDG2_L1[iz])
            np.fill_diagonal(kernel_4h_3111_iz, squeezed_4h_3111_iz)
            kernel_4h_3111[:, :, iz] = kernel_4h_3111_iz
        T_3111 = 12 * self.Pk[:, None, None]**2 * self.Pk[None, :, None] * kernel_4h_3111_iz * D[None, None, :]**6
        T_3111 += np.transpose(T_3111, (1,0,2))

        # 2211 Terms
        gamma, coef = self.fftLog_Pofk.get_power_and_coef()

        kernel_4h_2211_A = np.empty((kl, kl, zl)) * u.Mpc**3
        kernel_4h_2211_X = np.empty((kl, kl, zl)) * u.Mpc**3
        for iz, zi in enumerate(z):
            kernel_4h_2211_A_iz = 0.0
            squeezed_4h_2211_A_iz = 0.0
            kernel_4h_2211_X_iz = 0.0
            squeezed_4h_2211_X_iz = 0.0
            for gammai, coefi in zip(gamma, coef):
                kernel_4h_2211_A_iz += coefi * ingredients_T0.T2211_A_kernel(k[:,None], k[None,:], b1_L1[iz], b2_L1[iz], bG2_L1[iz], gammai)
                squeezed_4h_2211_A_iz += coefi * ingredients_T0.T2211_A_squeezed(k, b1_L1[iz], b2_L1[iz], bG2_L1[iz], gammai)
                kernel_4h_2211_X_iz += coefi * ingredients_T0.T2211_X_kernel(k[:,None], k[None,:], b1_L1[iz], b2_L1[iz], bG2_L1[iz], gammai)
                squeezed_4h_2211_X_iz += coefi * ingredients_T0.T2211_X_squeezed(k, b1_L1[iz], b2_L1[iz], bG2_L1[iz], gammai)
            np.fill_diagonal(kernel_4h_2211_A_iz, squeezed_4h_2211_A_iz)
            np.fill_diagonal(kernel_4h_2211_X_iz, squeezed_4h_2211_X_iz)
            kernel_4h_2211_A[:,:,iz] = kernel_4h_2211_A_iz.real * u.Mpc**3
            kernel_4h_2211_X[:,:,iz] = kernel_4h_2211_X_iz.real * u.Mpc**3

        T_2211_A = 8 * self.Pk[:, None, None]**2 * kernel_4h_2211_A * D[None, None, :]**6
        T_2211_A += np.transpose(T_2211_A, (1,0,2))
        T_2211_X = 16 * self.Pk[:, None, None] * self.Pk[None, :, None] * kernel_4h_2211_X * D[None, None, :]**6

        T_4h = I1[:, None, :]**2 * I1[None, :, :]**2 * (T_3111 + T_2211_X + T_2211_A)
        return T_4h

    def integrate_3h(self, z=None):
        k = self.k
        if z is None:
            z = self.z
        z = np.atleast_1d(z)
        D = np.atleast_1d(self.cosmo.growth_factor(1e-4*u.Mpc**-1, z, tracer=self.tracer))

        I1 = restore_shape(self.astro.Thalo(z, k, p=1), k, z)
        I2 = restore_shape(self.astro.Thalo(z, k, k, p=1), k, k, z)

        #This is the normalisation found in pyCCL, Not necessarily true for LIM
        I1 *= 1 / I1.value[0, :]
        I2 *= 1 / I1.value[0, :]**2

        b1_L1 =   np.atleast_1d(self.astro.bavg("b1", z, 1))
        b2_L1 =   np.atleast_1d(self.astro.bavg("b2", z, 1))
        bG2_L1 =  np.atleast_1d(self.astro.bavg("bG2", z, 1))
        b1_L2 =  np.atleast_1d(self.astro.bavg("b1", z, 2))
        b2_L2 =  np.atleast_1d(self.astro.bavg("b2", z, 2))
        bG2_L2 = np.atleast_1d(self.astro.bavg("bG2", z, 2))

        kl = len(k)
        zl = len(z)

        kernel_3h_211_A = ingredients_T0.T211_A_kernel(b1_L1, b1_L2, b2_L2, bG2_L2)
        T_211_A = kernel_3h_211_A * self.Pk[:, None, None] * self.Pk[None, :, None] * D[None, None, :]**4

        gamma, coef = self.fftLog_Pofk.get_power_and_coef()

        kernel_3h_221_X = np.empty((kl, kl, zl)) * u.Mpc**3
        for iz, zi in enumerate(z):
            kernel_3h_211_X_iz = 0.0
            squeezed_3h_211_X_iz = 0.0 
            for gammai, coefi in zip(gamma, coef):
                kernel_3h_211_X_iz += coefi * ingredients_T0.T211_X_kernel(k[:,None], k[None, :], b1_L1[iz], b2_L1[iz], bG2_L1[iz], b1_L2[iz], gammai)
                squeezed_3h_211_X_iz += coefi * ingredients_T0.T211_X_squeezed(k, b1_L1[iz], b2_L1[iz], bG2_L1[iz], b1_L2[iz], gammai)
            np.fill_diagonal(kernel_3h_211_X_iz, squeezed_3h_211_X_iz)
            kernel_3h_221_X[:, :, iz] = kernel_3h_211_X_iz.real * u.Mpc**3
        T_211_X =  kernel_3h_221_X * self.Pk[:, None, None] * D[None, None, :]**4
        T_211_X += np.transpose(T_211_X, (1, 0, 2))

        T_3h = 4 * I2 * I1[:, None, :] * I1[None, :, :] * (T_211_A + T_211_X)
        return T_3h

    def integrate_2h(self, z=None):
        k = self.k
        if z is None:
            z = self.z
        z = np.atleast_1d(z)
        D = np.atleast_1d(self.cosmo.growth_factor(1e-4*u.Mpc**-1, z, tracer=self.tracer))

        I1 = restore_shape(self.astro.Thalo(z, k, p=1), k, z)
        I2 = restore_shape(self.astro.Thalo(z, k, k, p=1), k, k, z)
        I3 = restore_shape(self.astro.Thalo(z, k, k, p=1, scale=(2,1)), k, k, z)

        #This is the normalisation found in pyCCL, Not necessarily true for LIM
        I1 *= 1 / I1.value[0, :]
        I2 *= 1 / I1.value[0, :]**2
        I3 *= 1 / I1.value[0, :]**3

        b1_L1 = np.atleast_1d(self.astro.bavg("b1", z, 1))
        b1_L2 = np.atleast_1d(self.astro.bavg("b1", z, 2))
        b1_L3 = np.atleast_1d(self.astro.bavg("b1", z, 3))

        kl = len(k)
        zl = len(z)

        kernel_2h_31 = np.empty(kl, iz)
        for iz, zi in enumerate(z):
            kernel_2h_31[:, iz] = (
                vZ1(b1_L1[iz], 0.0, 0.0, 0.0, 0.0)
                * vZ1(b1_L3[iz], 0.0, 0.0, 0.0, 0.0)
            )
        T_31 = 2 * self.Pk[None, :, None] * kernel_2h_31 * D[None, None, :]**2

        gamma, coef = self.fftLog_Pofk.get_power_and_coef()

        kernel_2h_22 = np.empty((kl, kl, zl)) * u.Mpc**3
        for iz, zi in enumerate(z):
            kernel_2h_22_iz = 0.0
            for gammai, coefi in zip(gamma, coef):
                kernel_2h_22_iz += coefi * ingredients_T0.T22_kernel(k[:,None], k[None,:], b1_L2, gammai)
            kernel_2h_22[:,:,iz] = kernel_2h_22_iz.real * u.Mpc**3
        T_22 = 2 * kernel_2h_22 * D[None, None, :]**2

        T_2h = (T_22 * I2**2
                + T_31 * I1[None, :, :] * I3
                + np.transpose(T_31 * I1[None, :, :] * I3, (1, 0, 2))
        )
        return T_2h

    def integrate_1h(self, z=None):
        k = self.k
        if z is None:
            z = self.z

        I1 = restore_shape(self.astro.Thalo(z, k, p=1).value, k, z)
        I4 = restore_shape(self.astro.Thalo(z, k, k, p=2, scale=(2,2)), k, k, z)

        #This is the normalisation found in pyCCL, Not necessarily true for LIM
        I4 *= 1 / I1.value[0, :]**4

        T_1h = I4

        return T_1h

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
