import numpy as np
import astropy.units as u
from copy import copy
from scipy.special import spherical_jn
from scipy.integrate import simpson

from SSLimPy.cosmology.cosmology import CosmoFunctions
from SSLimPy.utils.utils import *

class SurveySpecifications:
    def __init__(self, obspars, cosmo: CosmoFunctions):
        obspars = copy(obspars)
        obspars.setdefault("Tsys_NEFD", 40 * u.uK)
        obspars.setdefault("Nfeeds", 19)
        obspars.setdefault("beam_FWHM", 4.1 * u.arcmin)
        obspars.setdefault("nu", 115 * u.GHz)
        obspars.setdefault("dnu", 15 * u.MHz)
        obspars.setdefault("nuObs", 30 * u.GHz)
        obspars.setdefault("Delta_nu", 8 * u.GHz)
        obspars.setdefault("tobs", 1300 * u.h)
        obspars.setdefault("nD", 1)
        obspars.setdefault("Omega_field", 4 * u.deg**2)
        obspars.setdefault("N_FG_par", 1)
        obspars.setdefault("N_FG_perp", 1)
        obspars.setdefault("do_FG_wedge", False)
        obspars.setdefault("a_FG", 0.0)
        obspars.setdefault("b_FG", 0.0)

        self.obsparams = obspars
        self.cosmology = cosmo

    #####################
    # Survey Resolution #
    #####################
    # Call these functions before applying the AP transformations, scale fixes, ect

    def sigma_parr(self, z, nu_obs):
        x = (self.obsparams["dnu"] / nu_obs).to(1).value
        y = (1 + z) / self.cosmology.Hubble(z)
        return x * y

    def sigma_perp(self, z):
        x = self.obsparams["beam_FWHM"].to(u.rad).value / np.sqrt(8 * np.log(2))
        y = self.cosmology.angdist(z) * (1 + z)
        return x * y

    def F_parr(self, k, mu, z, nu_obs):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)

        logF = -0.5 * np.power(
            k[:, None, None]
            * mu[None, :, None]
            * self.sigma_parr(z, nu_obs)[None, None, :],
            2,
        )

        return np.squeeze(np.exp(logF))

    def F_perp(self, k, mu, z):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)

        logF = -0.5 * np.power(
            k[:, None, None]
            * np.sqrt(1 - mu[None, :, None] ** 2)
            * self.sigma_perp(z)[None, None, :],
            2,
        )

        return np.squeeze(np.exp(logF))

    ##################################
    # Convolution and Survey Windows #
    ##################################

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
        nu = self.obsparams["nu"]
        nuObs = self.obsparams["nuObs"]
        Delta_nu = self.obsparams["Delta_nu"]
        z = (nu / nuObs - 1).to(1).value
        z_min = (nu / (nuObs + Delta_nu / 2) - 1).to(1).value
        z_max = (nu / (nuObs - Delta_nu / 2) - 1).to(1).value

        # Construct W_survey (now just a cylinder)
        Sfield = self.Sfield(z, self.obsparams["Omega_field"])
        Lperp = np.sqrt(Sfield / np.pi)
        Lparr = self.Lfield(z_min, z_max)
        x = (qperp * Lperp).to(1).value
        Wperp = 2 * np.pi * Lperp * np.reshape(smooth_W(x.flatten()), x.shape) / qperp
        Wparr = Lparr * spherical_jn(0, (qparr * Lparr / 2).to(1).value)

        Wsurvey = Wperp * Wparr
        Vsurvey = (
            simpson(
                y=q[:, None] ** 3
                * simpson(y=np.abs(Wsurvey) ** 2 / (2 * np.pi) ** 2,x=muq, axis=1),
                x=np.log(q.value),
                axis=0,
            )
            * (Sfield * Lparr).unit
        )
        return Wsurvey, Vsurvey