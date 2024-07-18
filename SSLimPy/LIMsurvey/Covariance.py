import sys
from copy import copy
from time import time

import numpy as np
from astropy import constants as c
from astropy import units as u
from numba import njit, prange
from scipy.special import j1, spherical_jn
from scipy.integrate import simpson

from SSLimPy.interface import config as cfg


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
        Lperp = np.sqrt(Sfield / np.pi)
        Lparr = self.Lfield(z_min, z_max)
        Wperp = 2 * np.pi * Lperp * j1((qperp * Lperp).to(1).value) / qperp
        Wparr = Lparr * spherical_jn(0, (qparr * Lparr / 2).to(1).value)

        Wsurvey = Wperp * Wparr
        Vsurvey = simpson(q[:,None]**3*simpson(np.abs(Wsurvey)**2, muq, axis=1), np.log(q.value), axis=0)
        return Wsurvey, Vsurvey

    def convolved_Pk(self):
        """Convolves the Observed power spectrum with the survey volume

        This enters as the gaussian term in the final covariance
        """
        # Extract the pre-computed power spectrum
        k = self.powerspectrum.k
        mu = self.powerspectrum.mu
        Pobs = self.powerspectrum.Pk_Obs
        nz = Pobs.shape[-1]

        # Downsample q, muq and deltaphi
        nq = np.uint8(len(k) / cfg.settings["downsample_conv_q"])
        if "log" in cfg.settings["k_kind"]:
            q = np.geomspace(k[0], k[-1], nq)
        else:
            q = np.linspace(k[0], k[-1], nq)

        nmuq = np.uint8((len(mu)) / cfg.settings["downsample_conv_muq"])
        nmuq = nmuq + 1 - nmuq % 2
        muq = np.linspace(-1, 1, nmuq)
        muq = (muq[1:] + muq[:-1]) / 2.0
        deltaphi = np.linspace(-np.pi, np.pi, 2 * len(muq))

        # Obtain survey Window
        Wsurvey, Vsurvey = self.Wsurvey(q, muq)

        Pconv = np.empty(Pobs.shape)
        # Do the convolution for each redshift bin
        for iz in range(nz):
            Pconv[..., iz] = convolve(
                k.value,
                mu,
                q.value,
                muq,
                deltaphi,
                Pobs[:, :, iz].value,
                Wsurvey[:, :, iz].value,
            )
        return Pconv / Vsurvey


#########################
# Getting Jitty with it #
#########################


@njit(
    "(float64[::1], float64[::1], float64[:,:], float64[::1], float64[::1])",
    fastmath=True,
)
def _bilinear_interpolate(xi, yj, zij, x, y):
    # Check input sizes
    xl, yl = zij.shape
    rxl = x.size
    ryl = y.size
    assert xl == xi.size, "xi should be the same size as axis 0 of zij"
    assert yl == yj.size, "yj should be the same size as axis 1 of zij"
    assert ryl == rxl, "for every x should be a y"

    # Find the indices of the grid points surrounding (xi, yi)
    # Handle linear extrapolation for larger x,y
    x1_idx = np.searchsorted(xi, x)
    x1_idx[np.where(x1_idx == 0)] = 1
    x1_idx[np.where(x1_idx == xl)] = xl - 1
    x2_idx = x1_idx - 1

    y1_idx = np.searchsorted(yj, y)
    y1_idx[np.where(y1_idx == 0)] = 1
    y1_idx[np.where(y1_idx == yl)] = yl - 1
    y2_idx = y1_idx - 1

    # Get the coordinates of the grid points
    x1, x2 = xi[x1_idx], xi[x2_idx]
    y1, y2 = yj[y1_idx], yj[y2_idx]

    results = np.empty(rxl)
    for i in range(rxl):
        # Get the values at the grid points
        Q11 = zij[x1_idx[i], y1_idx[i]]
        Q21 = zij[x2_idx[i], y1_idx[i]]
        Q12 = zij[x1_idx[i], y2_idx[i]]
        Q22 = zij[x2_idx[i], y2_idx[i]]

        results[i] = (
            Q11 * (x2[i] - x[i]) * (y2[i] - y[i])
            + Q21 * (x[i] - x1[i]) * (y2[i] - y[i])
            + Q12 * (x2[i] - x[i]) * (y[i] - y1[i])
            + Q22 * (x[i] - x1[i]) * (y[i] - y1[i])
        ) / ((x2[i] - x1[i]) * (y2[i] - y1[i]))
    return results


# The numba trapz for phi, muq, and q
@njit("(float64[::1], float64[::1])", fastmath=True)
def _trapezoid(y, x):
    s = 0.0
    for i in range(x.size - 1):
        dx = x[i + 1] - x[i]
        dy = y[i] + y[i + 1]
        s += dx * dy
    return s * 0.5


@njit(
    "(float64[::1], float64[::1], "
    + "float64[::1], float64[::1], float64[::1], "
    + "float64[:,:], float64[:,:])",
    parallel=True,
)
def convolve(k, mu, q, muq, deltaphi, P, W):
    # Check input sizes
    kl, mul = P.shape
    assert kl == k.size, "k should be the same size as axis 0 of P"
    assert mul == mu.size, "mu should be the same size as axis 1 of P"
    ql, muql = W.shape
    deltaphil = deltaphi.size
    assert ql == q.size, "q should be the same size as axis 0 of W"
    assert muql == muq.size, "muq should be the same size as axis 1 of W"

    # create Return array to be filled in parallel
    Pconv = np.empty_like(P, dtype=np.float64)
    for ik in prange(kl):
        for imu in prange(mul):
            # use q, muq, deltaphi and obtain abs k-q and the polar angle of k-q
            abskminusq = np.empty((ql, muql, deltaphil))
            mukminusq = np.empty((ql, muql, deltaphil))
            for iq in range(ql):
                for imuq in range(muql):
                    for ideltaphi in range(deltaphil):
                        abskminusq[iq, imuq, ideltaphi] = np.sqrt(
                            np.power(k[ik], 2)
                            + np.power(q[iq], 2)
                            - 2
                            * q[iq]
                            * k[ik]
                            * (
                                muq[imuq] * mu[imu]
                                + np.sqrt(1 - np.power(muq[imuq], 2))
                                * np.sqrt(1 - np.power(mu[imu], 2))
                                * np.cos(deltaphi[ideltaphi])
                            )
                        )
                        mukminusq[iq, imuq, ideltaphi] = (
                            k[ik] * mu[imu] - q[iq] * muq[imuq]
                        ) / abskminusq[iq, imuq, ideltaphi]

            # flatten the axis last axis first
            abskminusq = abskminusq.flatten()
            mukminusq = mukminusq.flatten()
            # interpolate the logP on mu logk and fill with new values
            logPkminusq = _bilinear_interpolate(
                np.log(k), mu, np.log(P), np.log(abskminusq), mukminusq
            )
            logPkminusq = np.reshape(logPkminusq, (ql, muql, deltaphil))

            # Do the 3D trapezoid integration
            q_integrand = np.empty(ql)
            for iq in range(ql):
                muq_integrand = np.empty(muql)
                for imuq in range(muql):
                    phi_integrand = (
                        1
                        / (2 * np.pi) ** 3
                        * q[iq] ** 2
                        * (np.abs(W[iq, imuq]) ** 2)
                        * np.exp(logPkminusq[iq, imuq, :])
                    )
                    muq_integrand[imuq] = _trapezoid(phi_integrand, deltaphi)
                q_integrand[iq] = _trapezoid(muq_integrand, muq)
            Pconv[ik, imu] = _trapezoid(q_integrand * q, np.log(q))
    return Pconv
