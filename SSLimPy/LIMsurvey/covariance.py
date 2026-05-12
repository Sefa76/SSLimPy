import itertools
from functools import partial

import numpy as np
from astropy import units as u
from numba import njit, prange
from scipy.interpolate import UnivariateSpline as _UnivariateSpline
from scipy.integrate import trapezoid, simpson
from scipy.optimize import curve_fit
from scipy.special import legendre, roots_legendre
from py3nj import clebsch_gordan

from SSLimPy.LIMsurvey import power_spectrum
from SSLimPy.LIMsurvey.higher_order import *
from SSLimPy.LIMsurvey import ingredients_T0
from SSLimPy.utils.fft_log import FFTLog
from SSLimPy.utils.utils import *

UnivariateSpline = partial(_UnivariateSpline, s=0)


class Covariance:
    def __init__(self, power_spectrum: power_spectrum.PowerSpectra):
        self.cosmology = power_spectrum.fiducial_cosmology
        self.survey_specs = power_spectrum.survey_specs
        self.power_spectrum = power_spectrum
        self.settings = self.cosmology.settings
        self.k = self.power_spectrum.k
        self.dk = self.power_spectrum.dk
        self.mu = self.power_spectrum.mu
        self.z = self.power_spectrum.z

    def Nmodes(self):
        Vk = 4 * np.pi * self.k**2 * self.dk
        Vw = np.atleast_1d(self.survey_specs.Vfield())
        return Vk[:, None] * Vw[None, :] / ((2 * np.pi) ** 3)

    def get_detectornoise(self):
        PI = self.survey_specs.detector_noise()
        return PI.to(self.power_spectrum.Pk_Obs.unit)

    def compute_multipole_cov(self, sigma2):
        mu = self.mu
        if self.power_spectrum.mu_kind == "linear":
            def Pk_ell_moments(ell):
                L_ell = legendre(ell)(mu)
                return simpson(y=sigma2.value * L_ell[None, :, None], x=mu, axis=1)

        elif self.power_spectrum.mu_kind == "gauss":
            w = self.power_spectrum.w
            def Pk_ell_moments(ell):
                L_ell = legendre(ell)(mu)
                return np.sum(sigma2.value * L_ell[None, :, None] * w[None, :, None], axis=1)

        sigma_0 = Pk_ell_moments(0)
        sigma_2 = Pk_ell_moments(2)
        sigma_4 = Pk_ell_moments(4)
        sigma_6 = Pk_ell_moments(6)
        sigma_8 = Pk_ell_moments(8)

        cov_00 = 1 / 4 * sigma_0
        cov_20 = 5 / 4 * sigma_2
        cov_40 = 9 / 4 * sigma_4
        cov_22 = 25 / 4 * (
            clebsch_gordan(4, 4, 8, 0, 0, 0)**2 * sigma_4
            + clebsch_gordan(4, 4, 4, 0, 0, 0)**2 * sigma_2
            + clebsch_gordan(4, 4, 0, 0, 0, 0)**2 * sigma_0)
        cov_42 = 45 / 4 * (
            clebsch_gordan(8, 4, 12, 0, 0, 0)**2 * sigma_6
            + clebsch_gordan(8, 4, 8, 0, 0, 0)**2 * sigma_4
            + clebsch_gordan(8, 4, 4, 0, 0, 0)**2 * sigma_2
        )
        cov_44 = 81 / 4 * (
            clebsch_gordan(8, 8, 16, 0, 0, 0)**2 * sigma_8
            + clebsch_gordan(8, 8, 12, 0, 0, 0)**2 * sigma_6
            + clebsch_gordan(8, 8, 8, 0, 0, 0)**2 * sigma_4
            + clebsch_gordan(8, 8, 4, 0, 0, 0)**2 * sigma_2
            + clebsch_gordan(8, 8, 0, 0, 0, 0)**2 * sigma_0
        )

        # construct the covariance
        nk = np.uint16(len(self.k))
        nz = np.uint16(len(self.z))
        cov = construct_gaussian_cov(
            nk, nz, cov_00, cov_20, cov_40, cov_22, cov_42, cov_44
        )
        return cov * sigma2.unit

    def gaussian_nonoise_cov(self):
        Pobs = self.power_spectrum.Pk_Obs
        sigma2 = 2 * Pobs**2 / self.Nmodes()[:, None, :]
        return self.compute_multipole_cov(sigma2)

    def gaussian_cov(self):
        Pobs = self.power_spectrum.Pk_Obs
        PI = self.get_detectornoise()
        sigma2 = 2 * (Pobs + PI) ** 2 / self.Nmodes()[:, None, :]
        return self.compute_multipole_cov(sigma2)


class nonGuassianCov:
    def __init__(self, power_spectrum: power_spectrum.PowerSpectra):
        self.cosmo = power_spectrum.cosmology
        self.cfg = self.cosmo.cfg
        self.astro = power_spectrum.astro
        self.powerSpectrum = power_spectrum
        self.tracer = self.cfg.settings["TracerPowerSpectrum"]
        self.survey_specs = power_spectrum.survey_specs
        self.mu = power_spectrum.mu
        self.z = power_spectrum.z

        # Get power spectra on grids for numerical computations
        # TODO: For now only works for scale-independent growth
        self.k = power_spectrum.k
        self.Pk = self.cosmo.matpow(self.k, 0, nonlinear=False, tracer=self.tracer)
        self.kgrid = self.cosmo.k.to(u.Mpc**-1)
        self.Pgrid = self.cosmo.matpow(
            self.kgrid, 0.0, nonlinear=False, tracer=self.tracer
        ).to(u.Mpc**3)

        # FFTlog Approximation
        kmin_fftlog = self.cfg.settings["FFTlog_kmin"].to(u.Mpc**-1).value
        kmax_fftlog = self.cfg.settings["FFTlog_kmax"].to(u.Mpc**-1).value
        LogN = self.cfg.settings["FFTlog_LogN"]

        self.fftLog_Pofk = FFTLog(
            self.kgrid.value, self.Pgrid.value, kmin_fftlog, kmax_fftlog, LogN
        )

    def integrate_4h(self, z=None, return_ingredients=False):
        k = self.k
        if z is None:
            z = self.z
        z = np.atleast_1d(z)
        D = np.atleast_1d(
            self.cosmo.growth_factor(1e-4 * u.Mpc**-1, z, tracer=self.tracer)
        )

        I11 =   restore_shape(self.astro.Thalo(z, k, p=1, beta=1), k, z)
        I12 =   restore_shape(self.astro.Thalo(z, k, p=1, beta="b2_fitted"), k, z)
        I1G2 =  restore_shape(self.astro.Thalo(z, k, p=1, beta="bG2"), k, z)
        I13 =   restore_shape(self.astro.Thalo(z, k, p=1, beta="b3"), k, z)
        I1dG2 = restore_shape(self.astro.Thalo(z, k, p=1, beta="bdG2"), k, z)
        I1G3 =  restore_shape(self.astro.Thalo(z, k, p=1, beta="bG3"), k, z)
        I1DG2 = restore_shape(self.astro.Thalo(z, k, p=1, beta="bDG2"), k, z)

        kl = len(k)
        zl = len(z)

        # construct masks
        mask_k1ok2 = (k[:, None] / k[None, :]) < 5e-3
        mask_k2ok1 = (k[None, :] / k[:, None]) < 5e-3
        mask_diag = np.eye(kl, dtype=bool)
        mask_norm = ~np.logical_or(np.logical_or(mask_k1ok2, mask_k2ok1), mask_diag)
        mask_tril = np.logical_and(
            np.tril(np.ones((kl, kl), dtype=bool), k=-1), ~mask_k2ok1
        )

        # 3111 Terms
        T_3111_masks = [mask_k1ok2, mask_k2ok1, mask_norm]
        T_3111_funcs = [
            ingredients_T0.T3111_s_k1ok2,
            ingredients_T0.T3111_s_k2ok1,
            ingredients_T0.T3111_kernel,
        ]
        kernel_4h_3111 = np.empty((kl, kl, zl)) * u.uK**4

        for iz, zi in enumerate(z):
            I11_iz = I11[:, iz]
            I12_iz = I12[:, iz]
            I1G2_iz = I1G2[:, iz]
            I13_iz = I13[:, iz]
            I1dG2_iz = I1dG2[:, iz]
            I1G3_iz = I1G3[:, iz]
            I1DG2_iz = I1DG2[:, iz]

            for mask, func in zip(T_3111_masks, T_3111_funcs):
                i, j = np.where(mask)
                kernel_4h_3111[mask, iz] = func(
                    k[i],
                    k[j],
                    I11_iz[i],
                    I11_iz[j],
                    I12_iz[j],
                    I1G2_iz[j],
                    I13_iz[j],
                    I1dG2_iz[j],
                    I1G3_iz[j],
                    I1DG2_iz[j],
                )
            kernel_4h_3111[mask_diag, iz] = ingredients_T0.T3111_squeezed(
                k, I11_iz, I12_iz, I1G2_iz, I13_iz, I1dG2_iz, I1G3_iz, I1DG2_iz
            )
        T_3111 = (
            12
            * self.Pk[:, None, None] ** 2
            * self.Pk[None, :, None]
            * kernel_4h_3111
            * D[None, None, :] ** 6
        )
        T_3111 += np.transpose(T_3111, (1, 0, 2))

        # 2211 Terms
        gamma, coef = self.fftLog_Pofk.get_power_and_coef()

        T_2211_A_masks = [mask_k1ok2, mask_k2ok1, mask_norm]
        T_2211_A_funcs = [
            ingredients_T0.T2211_A_s_k1ok2,
            ingredients_T0.T2211_A_s_k2ok1,
            ingredients_T0.T2211_A_kernel,
        ]
        kernel_4h_2211_A = np.zeros((kl, kl, zl)) * u.Mpc**3 * u.uK**4

        T_2211_X_masks = [mask_k2ok1, mask_tril]
        T_2211_X_funcs = [
            ingredients_T0.T2211_X_s_k2ok1,
            ingredients_T0.T2211_X_kernel,
        ]
        kernel_4h_2211_X = np.zeros((kl, kl, zl)) * u.Mpc**3 * u.uK**4

        for iz, zi in enumerate(z):
            I11_iz = I11[:, iz]
            I12_iz = I12[:, iz]
            I1G2_iz = I1G2[:, iz]
            for gammai, coefi in zip(gamma, coef):

                A_iz = np.empty((kl, kl), dtype=complex) * u.uK**4
                for mask, func in zip(T_2211_A_masks, T_2211_A_funcs):
                    i, j = np.where(mask)
                    A_iz[mask] = func(
                        k[i],
                        k[j],
                        I11_iz[i],
                        I11_iz[j],
                        I12_iz[j],
                        I1G2_iz[j],
                        gammai,
                    )
                A_iz[mask_diag] = ingredients_T0.T2211_A_squeezed(
                    k, I11_iz, I12_iz, I1G2_iz, gammai
                )
                kernel_4h_2211_A[:, :, iz] += (coefi * A_iz).real * u.Mpc**3

                X_iz = np.zeros((kl, kl), dtype=complex) * u.uK**4
                for mask, func in zip(T_2211_X_masks, T_2211_X_funcs):
                    i, j = np.where(mask)
                    X_iz[mask] = func(
                        k[i],
                        k[j],
                        I11_iz[i],
                        I12_iz[i],
                        I1G2_iz[i],
                        I11_iz[j],
                        I12_iz[j],
                        I1G2_iz[j],
                        gammai,
                    )
                X_iz[mask_diag] = ingredients_T0.T2211_X_squeezed(
                    k, I11_iz, I12_iz, I1G2_iz, gammai
                )  # / 2 # The factor 1/2 is for the mirroring of the full expression
                kernel_4h_2211_X[:, :, iz] += (coefi * X_iz).real * u.Mpc**3
        kernel_4h_2211_X += kernel_4h_2211_X.transpose((1, 0, 2))

        T_2211_A = (
            8 * self.Pk[:, None, None] ** 2 * kernel_4h_2211_A * D[None, None, :] ** 6
        )
        T_2211_A += np.transpose(T_2211_A, (1, 0, 2))
        T_2211_X = (
            16
            * self.Pk[:, None, None]
            * self.Pk[None, :, None]
            * kernel_4h_2211_X
            * D[None, None, :] ** 6
        )

        if return_ingredients:
            return T_3111, T_2211_X, T_2211_A
        T_4h = T_3111 + T_2211_X + T_2211_A
        return T_4h

    def integrate_3h(self, z=None):
        k = self.k
        if z is None:
            z = self.z
        z = np.atleast_1d(z)
        D = np.atleast_1d(
            self.cosmo.growth_factor(1e-4 * u.Mpc**-1, z, tracer=self.tracer)
        )

        I11 =   restore_shape(self.astro.Thalo(z, k, p=1, beta=1), k, z)
        I12 =   restore_shape(self.astro.Thalo(z, k, p=1, beta="b2_fitted"), k, z)
        I1G2 =  restore_shape(self.astro.Thalo(z, k, p=1, beta="bG2"), k, z)
        I21 =   restore_shape(self.astro.Thalo(z, k, p=1, scale=(2,), beta=1), k, k, z)
        I22 =   restore_shape(self.astro.Thalo(z, k, p=1, scale=(2,), beta="b2_fitted"), k, k, z)
        I2G2 =  restore_shape(self.astro.Thalo(z, k, p=1, scale=(2,), beta="bG2"), k, k, z)

        kl = len(k)
        zl = len(z)

        kernel_3h_211_A = ingredients_T0.T211_A_kernel(
            I11[:, None, :], I11[None, :, :], I21, I22, I2G2
        )
        T_211_A = (
            kernel_3h_211_A
            * self.Pk[:, None, None]
            * self.Pk[None, :, None]
            * D[None, None, :] ** 4
        )

        gamma, coef = self.fftLog_Pofk.get_power_and_coef()

        kernel_3h_221_X = np.empty((kl, kl, zl)) * u.uK**4 * u.Mpc**6
        for iz, zi in enumerate(z):
            kernel_3h_211_X_iz = 0.0
            squeezed_3h_211_X_iz = 0.0

            I11_iz = I11[:, iz]
            I12_iz = I12[:, iz]
            I1G2_iz = I1G2[:, iz]
            I21_iz = I21[:, :, iz]
            for gammai, coefi in zip(gamma, coef):
                kernel_3h_211_X_iz += coefi * ingredients_T0.T211_X_kernel(
                    k[:, None],
                    k[None, :],
                    I11_iz[:, None],
                    I12_iz[:, None],
                    I1G2_iz[:, None],
                    I21_iz,
                    gammai,
                )
                squeezed_3h_211_X_iz += coefi * ingredients_T0.T211_X_squeezed(
                    k, I11_iz, I12_iz, I1G2_iz, np.diag(I21_iz), gammai
                )
            np.fill_diagonal(kernel_3h_211_X_iz, squeezed_3h_211_X_iz)
            kernel_3h_221_X[:, :, iz] = kernel_3h_211_X_iz.real * u.Mpc**3
        T_211_X = kernel_3h_221_X * self.Pk[:, None, None] * D[None, None, :] ** 4
        T_211_X += np.transpose(T_211_X, (1, 0, 2))

        T_3h = 4 * (T_211_A + T_211_X)
        return T_3h

    def integrate_2h(self, z=None):
        k = self.k
        if z is None:
            z = self.z
        z = np.atleast_1d(z)
        D = np.atleast_1d(
            self.cosmo.growth_factor(1e-4 * u.Mpc**-1, z, tracer=self.tracer)
        )

        I1 = restore_shape(self.astro.Thalo(z, k, p=1, beta=1), k, z)
        I2 = restore_shape(self.astro.Thalo(z, k, k, p=2, beta=1), k, k, z)
        I3 = restore_shape(
            self.astro.Thalo(z, k, k, p=2, scale=(2, 1), beta=1), k, k, z
        )

        kl = len(k)
        zl = len(z)

        T_31 = 2 * self.Pk[None, :, None] * I1[None, :, :] * I3 * D[None, None, :] ** 2
        T_31 += np.transpose(T_31, (1, 0, 2))

        gamma, coef = self.fftLog_Pofk.get_power_and_coef()
        kernel_2h_22 = np.empty((kl, kl, zl)) * u.Mpc**3
        for iz, zi in enumerate(z):
            kernel_2h_22_iz = 0.0
            for gammai, coefi in zip(gamma, coef):
                kernel_2h_22_iz += coefi * ingredients_T0.T22_kernel(
                    k[:, None], k[None, :], gammai
                )
            kernel_2h_22[:, :, iz] = kernel_2h_22_iz.real * u.Mpc**3
        T_22 = 2 * kernel_2h_22 * I2**2 * D[None, None, :] ** 2

        T_2h = T_22 + T_31
        return T_2h

    def integrate_1h(self, z=None):
        k = self.k
        if z is None:
            z = self.z

        I4 = restore_shape(self.astro.Thalo(z, k, k, p=2, scale=(2, 2)), k, k, z)

        T_1h = I4

        return T_1h

    def compute_nG_Cov(self):
        T_1h = self.integrate_1h()
        T_2h = self.integrate_2h()
        T_3h = self.integrate_3h()
        T_4h = self.integrate_4h()

        V = self.survey_specs.Vfield()
        return (T_1h + T_2h + T_3h + T_4h) / V


class SuperSampleCovariance:
    def __init__(self, power_spectrum: power_spectrum.PowerSpectra):
        self.power_spectrum = power_spectrum
        self.cosmology = power_spectrum.cosmology
        self.halomodel = power_spectrum.halomodel
        self.astro = power_spectrum.astro
        self.survey_specs = power_spectrum.survey_specs

        self.k = power_spectrum.k
        self.mu = power_spectrum.mu
        self.kgrid = power_spectrum.k_numerics
        self.z = power_spectrum.z

    def sigma_survey(self):
        k = self.kgrid
        mu = self.mu
        z = np.atleast_1d(self.z)

        V = self.survey_specs.Vfield()
        W = (self.survey_specs.Wsurvey(self.kgrid, self.mu) / V).to(1).value
        W = np.reshape(W, (*k.shape, *mu.shape, *z.shape))

        P = np.reshape(
            self.cosmology.matpow(k, z, nonlinear=False, tracer=self.halomodel.tracer),
            (*k.shape, *z.shape),
        )
        D = (4 * np.pi * (self.kgrid[:, None, None] / (2 * np.pi))**3 * P[:, None, :]).to(1).value
        sigma2_intgrnd = D * W**2

        sigma2_intgrnd = np.trapezoid(sigma2_intgrnd, x=mu, axis=1) / 2
        sigma2 = np.trapezoid(sigma2_intgrnd, x=np.log(k.value), axis=0)
        return sigma2.squeeze()

    def halo_sample_variance(self, k, z):
        """The standard  result for halo sample variance"""
        k = np.atleast_1d(k)
        z = np.atleast_1d(z)

        b1_L2 = np.reshape(
            self.astro.Thalo(z, k, p=1, scale=(2,), beta=1),
            (*k.shape, *z.shape),
        )

        return b1_L2

    def linear_growth_response(self, k, z):
        """Corresponds to P * C21 in Wadekar et al."""
        k = np.atleast_1d(k).to(self.kgrid.unit)
        z = np.atleast_1d(z)

        Pk = np.reshape(
            self.cosmology.matpow(
                self.kgrid, z, tracer=self.power_spectrum.tracer, nonlinear=False
            ),
            (*self.kgrid.shape, *z.shape),
        )
        Pk_response = np.reshape(
            self.cosmology.matpow(
                k, z, tracer=self.power_spectrum.tracer, nonlinear=False
            ),
            (*k.shape, *z.shape),
        )

        Delta = (4 * np.pi * (self.kgrid[:, None] / 2 / np.pi) ** 3 * Pk).to(1).value

        gamma = []
        for iz, zi in enumerate(z):
            gamma.append(
                UnivariateSpline(
                    np.log(self.kgrid.value),
                    np.log(Delta[:, iz]),
                ).derivative(1)(
                    np.log(k.value),
                )
            )
        gamma = np.array(gamma).T

        b1_L1 = np.reshape(
            self.astro.Thalo(z, k, p=1, scale=(1,), beta=1), (*k.shape, *z.shape)
        )
        return b1_L1**2 * Pk_response * (68 / 21 - gamma / 3)

    def biased_clustering_response(self, k, z):
        """Corresponds to the spherical average second order bias"""
        k = np.atleast_1d(k)
        z = np.atleast_1d(z)

        Pk = np.reshape(
            self.cosmology.matpow(
                k, z, tracer=self.power_spectrum.tracer, nonlinear=False
            ),
            (*k.shape, *z.shape),
        )

        b1_L1 = np.reshape(
            self.astro.Thalo(z, k, p=1, scale=(1,), beta=1), (*k.shape, *z.shape)
        )
        # b2_L1 = np.reshape(
        #     self.astro.Thalo(z, k, p=1, scale=(1,), beta="b2"), (*k.shape, *z.shape)
        # )
        # bG2_L1 = np.reshape(
        #     self.astro.Thalo(z, k, p=1, scale=(1,), beta="bG2"), (*k.shape, *z.shape)
        # )
        # bsph = b2_L1 - 4 / 3 * bG2_L1

        bsph = np.reshape(
            self.astro.Thalo(z, k, p=1, scale=(1,), beta="b2sph_lazeyras"), (*k.shape, *z.shape)
        )

        return 2 * b1_L1 * bsph * Pk

    def response(self, k, z):
        response = (
            self.linear_growth_response(k, z)
            + self.biased_clustering_response(k, z)
            + self.halo_sample_variance(k, z)
        )
        return np.squeeze(response)

    def logresponse(self, k, z):
        # TODO: Add proper handling of RSD in response functions
        assert self.cosmology.settings["do_RSD"] == False

        k = np.atleast_1d(k)
        z = np.atleast_1d(z)

        response = self.response(k, z).reshape((*k.shape, *z.shape))

        Pk_response = self.cosmology.matpow(
            k, z, tracer=self.power_spectrum.tracer, nonlinear=False,
            ).reshape(*k.shape, *z.shape)
        I11 = self.astro.Thalo(z, k, p=1, scale=(1,), beta=1).reshape((*k.shape, *z.shape))
        I02 = self.astro.Thalo(z, k, p=1, scale=(2,), beta=0).reshape((*k.shape, *z.shape))
        Pk_halo = I11**2 * Pk_response + I02

        return np.squeeze(response / Pk_halo)

    def compute_SSC(self):
        k = self.k
        z = self.z

        # It is done this way to approimate the effect of survey resolution etc as beeing independent of backgoundmodes
        P = self.power_spectrum.Pk_0bs.reshape((*k.shape, *z.shape))
        logR = self.logresponse(self.k, z).reshape((*k.shape, *z.shape))
        response = P * logR

        sigma = np.atleast_1d(self.sigma_survey())

        SSC = sigma[None, None, :] * response[:, None, :] * response[None, :, :]
        return SSC


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
