import astropy.units as u
import numpy as np
from numba import njit, prange

from SSLimPy.interface import config as cfg
from SSLimPy.utils.utils import linear_interpolate
from SSLimPy.cosmology.cosmology import cosmo_functions


# This class will contain relevant functions to compute
# ingredients for the different halo model contributions
class nonlinear(cosmo_functions):
    def __init__(
        self, tracer, cosmopars=dict(), nuiscance_like=dict(), input=None, cosmology=None
    ):
        super().__init__(cosmopars, nuiscance_like, input, cosmology)
        self.tracer = tracer
        self.R = np.geomspace(
            cfg.settings["Rmin"],
            cfg.settings["Rmax"],
            cfg.settings["nR"],
        ).to(u.Mpc)
        self.z = np.linspace(
            0,
            5,
            50,
        )
        self.k = np.geomspace(
            cfg.settings["kmin"],
            cfg.settings["kmax"],
            cfg.settings["nk"],
        )

        self.sigmaR_lut, self.dsigmaR_lut = self.create_sigmaR_lookuptable()

    def create_sigmaR_lookuptable(self):
        # Unitless linear power spectrum
        R = self.R
        z = self.z
        kinter = self.k.to(u.Mpc**-1)
        Dkinter = (
            (
                4
                * np.pi
                * (kinter[:, None] / (2 * np.pi)) ** 3
                * self.matpow(kinter, z, nonlinear="False", tracer=self.tracer)
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
            cfg.settings["nt"],
            cfg.settings["alpha_iSigma"],
        )
        return sigmaR, dsigmaR


##############
# Numba Part #
##############


@njit
def smooth_W(x):
    lx = len(x)
    W = np.empty(lx)
    for ix, xi in enumerate(x):
        if xi < 1e-3:
            W[ix] = 1. - 1. / 10. * xi**2
        else:
            W[ix] = 3. / xi**3. * (np.sin(xi) - xi * np.cos(xi))
    return W


@njit
def smooth_dW(x):
    lx = len(x)
    dW = np.empty(lx)
    for ix, xi in enumerate(x):
        if xi < 1e-3:
            dW[ix] = -1. / 5. * xi
        else:
            dW[ix] = 3. / xi**2. * np.sin(xi) - 9 / xi**4 * (np.sin(xi) - xi * np.cos(xi))
    return dW


@njit("(float64[::1], float64, float64[::1], float64[:], float64)", fastmath=True)
def sigma_integrand(t, R, kinter, Dkinter, alpha):
    nt = len(t)
    integrand = np.empty(nt)

    Rk = (1 / t[1:-1] - 1) ** alpha
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
    integrand[1:-1] = alpha * Dk * W**2 / (t[1:-1] * (1 - t[1:-1]))
    integrand[0], integrand[-1] = 0, 0
    return integrand


@njit("(float64[::1], float64, float64[::1], float64[:], float64)", fastmath=True)
def dsigma_integrand(t, R, kinter, Dkinter, alpha):
    nt = len(t)
    integrand = np.empty(nt)

    Rk = (1 / t[1:-1] - 1) ** alpha
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
    integrand[1:-1] = alpha * Dk * 2 * k * dW * W / (t[1:-1] * (1 - t[1:-1]))
    integrand[0], integrand[-1] = 0, 0
    return integrand


@njit(
    "(float64[::1], float64[::1], float64[::1], float64[:,:], uint64, float64)",
    parallel=True,
)
def sigmas_of_R_and_z(R, z, kinter, Dkinter, nt, alpha):
    Rl, zl = len(R), len(z)
    sigma = np.empty((Rl, zl))
    dsigma = np.empty((Rl, zl))

    # integration will be done on a log-spaced grid
    t = np.linspace(0., 1., nt)
    dt = 1 / (nt - 1)
    print((1 / t[1:-1] - 1) ** alpha)
    for iz in prange(zl):
        for iR in prange(Rl):
            integrand_sigma = sigma_integrand(t, R[iR], kinter, Dkinter[:, iz], alpha)
            integrand_dsigma = dsigma_integrand(t, R[iR], kinter, Dkinter[:, iz], alpha)
            sigma[Rl, zl] = np.sum(
                (integrand_sigma[1:] + integrand_sigma[:-1]) / 2 * dt
            )
            dsigma[Rl, zl] = np.sum(
                (integrand_dsigma[1:] + integrand_dsigma[:-1]) / 2 * dt
            )
    return sigma, dsigma

