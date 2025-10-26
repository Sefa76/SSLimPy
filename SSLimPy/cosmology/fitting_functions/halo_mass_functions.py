"""
Calculate the halo mass function as function of mass for different fitting functions

All functions return dn/dM(M)

Each function takes: self, Mvec, rhoM, z

Takes inspiration from pylians
"""

import numpy as np
from functools import partial
import astropy.units as u


class halo_mass_functions:
    def __init__(self, halomodel):
        self.halomodel = halomodel
        self.cosmology = halomodel.cosmology

        # SigmaM functions
        self.sigmaM = partial(self.halomodel.sigmaM, tracer=self.halomodel.tracer)
        self.dsigmaM_dM = partial(
            self.halomodel.dsigmaM_dM, tracer=self.halomodel.tracer
        )

    ###################################
    # Mass functions in natural units #
    ###################################

    def PS_nuf(self, sigma, z):
        """Press--Schechter function in natural units."""
        nu = self.halomodel.delta_crit / sigma

        f = np.sqrt(2.0 / np.pi) * nu * np.exp(-0.5 * nu**2)
        return f

    def ST_nuf(self, sigma, z):
        """Sheth--Torman functon in natural units."""
        nu = self.halomodel.delta_crit / sigma

        A = 0.3222
        a = 0.707
        p = 0.3

        nup = a * nu**2
        f = A * np.sqrt(nup * 2.0 / np.pi) * np.exp(-0.5 * nup) * (1.0 + 1.0 / nup**p)
        return f

    def Tinker_nuf(self, sigma, z):
        """Tinker et al. function in natural units."""
        sigma = np.atleast_1d(sigma)
        z = np.atleast_1d(z)

        Delta = 200
        alpha = 10 ** (-((0.75 / np.log10(Delta / 75.0)) ** 1.2))
        A = 0.186 * (1.0 + z) ** (-0.14)
        a = 1.47 * (1.0 + z) ** (-0.06)
        b = 2.57 * (1.0 + z) ** (-alpha)
        c = 1.19

        f = A * ((sigma / b) ** -a + 1.0) * np.exp(-c / sigma**2)
        return f

    def Crocce_nuf(self, sigma, z):
        """Corcce et al. function in natural units."""
        A = 0.58 * (1.0 + z) ** (-0.13)
        a = 1.37 * (1.0 + z) ** (-0.15)
        b = 0.3 * (1.0 + z) ** (-0.084)
        c = 1.036 * (1.0 + z) ** (-0.024)

        f = A * (sigma**-a + b) * np.exp(-c / sigma**2)
        return f

    def Jenkins_nuf(self, sigma, z):
        """Jenkins et al. function in natural units."""
        A = 0.315
        b = 0.61
        c = 3.8

        f = A * np.exp(-np.absolute(np.log(1.0 / sigma) + b) ** c)
        return f

    def Warren_nuf(self, sigma, z):
        """Warren et al. function in natural untis."""
        A = 0.7234
        a = 1.625
        b = 0.2538
        c = 1.1982

        f = A * (sigma ** (-a) + b) * np.exp(-c / sigma**2)
        return f

    def Watson_nuf(self, sigma, z):
        """Watson et al. function in natural units."""
        delta = 200.0
        OmegaM = self.cosmology.Omega(0, "matter")
        A = 0.194
        a = 1.805
        b = 2.267
        c = 1.287

        factor = (
            np.exp(0.023 * (delta / 178.0 - 1.0))
            * (delta / 178.0) ** (-0.456 * OmegaM - 0.139)
            * np.exp(0.072 * (1 - delta / 178.0) / sigma**2.130)
        )
        f = A * (sigma ** (-a) + b) * np.exp(-c / sigma**2) * factor
        return f

    def Watson_FOF_nuf(self, sigma, z):
        """Watson et al. function using FOF in natural units."""
        A = 0.282
        a = 2.163
        b = 1.406
        c = 1.210

        f = A * ((b / sigma) ** a + 1.0) * np.exp(-c / sigma**2)
        return f

    def Angulo_nuf(self, sigma, z):
        """Angulo et al. function in natural units."""
        f = 0.265 * (1.675 / sigma + 1.0) ** 1.9 * np.exp(-1.4 / sigma**2)
        return f

    ###############################
    # Conversion from nuf to dndM #
    ###############################

    def converter(self, M, z, fnu):
        """Converter function to go from nu f to dn_dM"""
        M = np.atleast_1d(M)
        z = np.atleast_1d(z)

        sigmaM = np.reshape(
            self.sigmaM(M, z),
            (*M.shape, *z.shape),
        )
        dsigmaM_dM = np.reshape(
            self.dsigmaM_dM(M, z),
            (*M.shape, *z.shape),
        )
        rhoM = self.halomodel.rho_tracer
        f = fnu(sigmaM, z)

        dn_dM = -f * rhoM / M[:, None] * dsigmaM_dM / sigmaM
        return np.squeeze(dn_dM).to(u.Mpc**-3 * u.Msun**-1)

    #################################
    # Converted halo mass functions #
    #################################

    def ST(self, M, z):
        """
        Sheth-Tormen halo mass function
        """
        return self.converter(M, z, self.ST_nuf)

    def Tinker(self, M, z):
        """
        Tinker et al 2008 halo mass function for delta=200
        """
        return self.converter(M, z, self.Tinker_nuf)

    def Crocce(self, M, z):
        """
        Crocce et al. halo mass function
        """
        return self.converter(M, z, self.Crocce_nuf)

    def Jenkins(self, M, z):
        """
        Jenkins et al. halo mass function
        """
        return self.converter(M, z, self.Jenkins_nuf)

    def Warren(self, M, z):
        """
        Warren et al. halo mass function
        """
        return self.converter(M, z, self.Warren_nuf)

    def Watson(self, M, z):
        """
        Watson et al. halo mass function for delta=200. Can be changed
        """
        return self.converter(M, z, self.Watson_nuf)

    def Watson_FOF(self, M, z):
        """
        Watson et al. halo mass function using FOF
        """
        return self.converter(M, z, self.Watson_FOF_nuf)

    def Angulo(self, M, z):
        """
        Angulo et al. halo mass function
        """
        return self.converter(M, z, self.Angulo_nuf)
