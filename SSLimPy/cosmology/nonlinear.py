import astropy.units as u
import numpy as np
from numba import njit, prange
from SSLimPy.cosmology.cosmology import cosmo_functions
from SSLimPy.interface import config as cfg
from SSLimPy.utils.utils import *


# This class will contain relevant functions to compute
# ingredients for the different halo model contributions
class nonlinear(cosmo_functions):
    def __init__(
        self,
        tracer,
        cosmopars=dict(),
        nuiscance_like=dict(),
        input_type=None,
        cosmology=None,
    ):
        super().__init__(cosmopars, nuiscance_like, input_type, cosmology)
        self.tracer = tracer
        self.R = np.geomspace(
            cfg.settings["Rmin"],
            cfg.settings["Rmax"],
            cfg.settings["nR"],
        ).to(u.Mpc)
        self.z = np.linspace(
            0,
            5,
            64,
        )
        self.k = np.geomspace(
            cfg.settings["kmin"],
            cfg.settings["kmax"],
            cfg.settings["nk"],
        ).to(u.Mpc**-1)

        self.sigmaR_lut, self.dsigmaR_lut = self._create_sigmaR_lookuptable()

    def _create_sigmaR_lookuptable(self):
        # Unitless linear power spectrum
        R = self.R
        z = self.z
        kinter = self.k.to(u.Mpc**-1)
        Dkinter = (
            (
                4
                * np.pi
                * (kinter[:, None] / (2 * np.pi)) ** 3
                * self.matpow(kinter, z, nonlinear=False, tracer=self.tracer)
            )
            .to(1)
            .value
        )

        # obtain sigma grids
        sigmaR, dsigmaR = sigmas_of_R_and_z(
            R.value,
            z,
            kinter.value,
            Dkinter,
            cfg.settings["alpha_iSigma"],
            cfg.settings["tol_sigma"],
        )
        return sigmaR, dsigmaR

    def read_lut(self, R, z, output="sigma"):
        """Method to read the sigmaR, dsigmaR look up tabels.
        Will do power law extrapolation.
        Returns dictionarry with desired output with combined shape of the inputs.
        """
        # Array preperation. We will do power law extrapolation in R[Mpc].
        Rs = np.atleast_1d(R).shape
        zs = np.atleast_1d(z).shape
        logR = np.log(np.atleast_1d(R).flatten().to(u.Mpc).value)
        z = np.atleast_1d(z).flatten()
        vlogR = np.repeat(logR, len(z))
        vz = np.tile(z, len(logR))

        mlogR = np.log(self.R.to(u.Mpc).value)
        mz = self.z
        result = dict()
        if "sigma" in output or "both" in output:
            sigma = np.exp(
                bilinear_interpolate(mlogR, mz, np.log(self.sigmaR_lut), vlogR, vz)
            )
            result["sigma"] = np.reshape(sigma, (*Rs, *zs))

        if "dsigma" in output or "both" in output:
            dsigma = (
                -np.exp(
                    bilinear_interpolate(
                        mlogR, mz, np.log(-self.dsigmaR_lut), vlogR, vz
                    )
                )
                * u.Mpc**-1
            )
            result["dsigma"] = np.reshape(dsigma, (*Rs, *zs))
        return result

    def sigmaR_of_z(self, R, z, tracer="matter"):
        if tracer == self.tracer:
            sigma = self.read_lut(R, z, output="sigma")["sigma"]
            return np.squeeze(sigma)
        else:
            return super().sigmaR_of_z(R, z, tracer)

    def dsigmaR_of_z(self, R, z, tracer="matter"):
        if tracer == self.tracer:
            dsigma = self.read_lut(R, z, output="dsigma")["dsigma"]
            return np.squeeze(dsigma)
        else:
            return super().dsigmaR_of_z(R, z, tracer)


##############
# Numba Part #
##############


@njit("(float64[::1], float64, float64[::1], float64[:], uint64)", fastmath=True)
def sigma_integrand(t, R, kinter, Dkinter, alpha):
    nt = len(t)
    integrand = np.zeros(nt)
    # mask out region where integrand is 0
    mask = (t > 0) & (t < 1)

    Rk = (1 / t[mask] - 1) ** alpha
    k = Rk / R
    W = smooth_W(Rk)

    # power law extrapolation
    Dk = np.exp(
        linear_interpolate(
            np.log(kinter),
            np.log(Dkinter),
            np.log(k),
        )
    )
    integrand[mask] = alpha * Dk * W**2 / (t[mask] * (1 - t[mask]))
    return integrand


@njit("(float64[::1], float64, float64[::1], float64[:], uint64)", fastmath=True)
def dsigma_integrand(t, R, kinter, Dkinter, alpha):
    nt = len(t)
    integrand = np.zeros(nt)
    # mask out region where integrand is 0
    mask = (t > 0) & (t < 1)
    Rk = (1 / t[mask] - 1) ** alpha
    k = Rk / R
    W = smooth_W(Rk)
    dW = smooth_dW(Rk)

    # power law extrapolation
    Dk = np.exp(
        linear_interpolate(
            np.log(kinter),
            np.log(Dkinter),
            np.log(k),
        )
    )
    integrand[mask] = alpha * Dk * 2 * k * dW * W / (t[mask] * (1 - t[mask]))
    return integrand


@njit
def adaptive_mesh_integral(a, b, integrand, R, kinter, Dkinter, alpha, eps):
    """Adapted from CAMB and Class implementation of HMCode2020 by Alexander Mead"""
    if a == b:
        return 0

    # Define the minimum and maximum number of iterations
    jmin = 5  # Minimum iterations to avoid premature convergence
    jmax = 20  # Maximum iterations before timeout

    # Initialize sum variables for integration
    sum_2n = 0.0
    sum_n = 0.0
    sum_old = 0.0
    sum_new = 0.0

    for j in range(1, jmax + 1):
        n = 1 + 2 ** (j - 1)
        dx = (b - a) / (n - 1)
        if j == 1:
            f1 = integrand(np.array([float(a)]), R, kinter, Dkinter, alpha)[0]
            f2 = integrand(np.array([float(b)]), R, kinter, Dkinter, alpha)[0]
            # print(f1, f2)
            sum_2n = 1.0 / 2.0 * (f1 + f2) * dx
            sum_new = sum_2n
        else:
            t = a + (b - a) * (np.arange(2, n, 2) - 1) / (n - 1)
            I = integrand(t, R, kinter, Dkinter, alpha)
            sum_2n = sum_n / 2 + np.sum(I) * dx
            sum_new = (4 * sum_2n - sum_n) / 3
            # print(sum_new, sum_old)

        if j >= jmin and sum_old != 0:
            if abs(1.0 - sum_new / sum_old) < eps:
                return sum_new
        if j == jmax:
            print("INTEGRATE: Integration timed out")
            return sum_new
        else:
            sum_old = sum_new
            sum_n = sum_2n
            sum_2n = 0.0
    return sum_new


@njit(
    "(float64[::1], float64[::1], float64[::1], float64[:,:], uint64, float64)",
    parallel=True,
)
def sigmas_of_R_and_z(R, z, kinter, Dkinter, alpha, eps):
    Rl, zl = len(R), len(z)
    sigma = np.empty((Rl, zl))
    dsigma = np.empty((Rl, zl))

    # integration will be done on a log-spaced grid in k from 0 to infinity via transformation
    for iz in prange(zl):
        for iR in prange(Rl):
            sintegral = adaptive_mesh_integral(
                0, 1, sigma_integrand, R[iR], kinter, Dkinter[:, iz], alpha, eps
            )
            sigma[iR, iz] = np.sqrt(sintegral)
            dsintegral = adaptive_mesh_integral(
                0, 1, dsigma_integrand, R[iR], kinter, Dkinter[:, iz], alpha, eps
            )
            dsigma[iR, iz] = dsintegral / (2 * sigma[iR, iz])
    return sigma, dsigma
