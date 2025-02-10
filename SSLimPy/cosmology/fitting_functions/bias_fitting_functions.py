"""
Calculate the halo bias depending on the mass for different fitting functions

All functions return b(M,z)

Bias model parameter values are given by a dictionary called bias_par.
Each function should have the same signature (M, z, dc) -> scalar or array like
"""

from copy import deepcopy
from functools import partial
import numpy as np


class bias_fitting_functions:

    def __init__(self, astro):
        self.astro = astro  # dont call it astrology
        self.cosmology = astro.cosmology
        self.astroparams = deepcopy(astro.astroparams)
        self.bias_par = self.astroparams["bias_par"]
        self.sigmaM = partial(self.astro.sigmaM, tracer=self.astro.astrotracer)
        self.dsigmaM_dM = partial(self.astro.dsigmaM_dM, tracer=self.astro.astrotracer)

    def Tinker10(self, M, z, dc):

        nu = dc / self.sigmaM(M, z)
        # Parameters of bias fit
        y = self.bias_par.get("y", np.log10(200.0))
        B = self.bias_par.get("B", 0.183)
        b = self.bias_par.get("b", 1.5)
        c = self.bias_par.get("c", 2.4)

        A = 1.0 + 0.24 * y * np.exp(-((4.0 / y) ** 4.0))
        C = 0.019 + 0.107 * y + 0.19 * np.exp(-((4.0 / y) ** 4.0))
        a = 0.44 * y - 0.88

        return 1.0 - A * nu**a / (nu**a + dc**a) + B * nu**b + C * nu**c

    def Mo96(self, M, z, dc):
        """
        Peak-background split bias correspdonding to PS HMF.

        Taken from Mo and White (1996)
        """
        nu = dc / self.sigmaM(M, z)

        return 1.0 + (nu**2.0 - 1.0) / dc

    def Jing98(self, M, z, dc):
        """
        Empirical bias of Jing (1998): http://adsabs.harvard.edu/abs/1998ApJ...503L...9J
        """
        M = np.atleast_1d(M)
        z = np.atleast_1d(z)

        M_NL = self.astro.mass_non_linear(z, delta_crit=dc)
        x = (M[:, None] / M_NL[None, :]).to(1).value

        ns = self.cosmology.input_cosmopars.get(
            "ns", self.cosmology.input_cosmopars["n_s"]
        )
        nu_star = x ** (ns + 3.0) / 6.0

        a = self.bias_par.get("a", 0.5)
        b = self.bias_par.get("b", 0.06)
        c = self.bias_par.get("c", 0.02)

        bh = (a / nu_star**4.0 + 1.0) ** (b - c * ns) * (
            1.0 + (nu_star**2.0 - 1.0) / dc
        )
        return np.squeeze(bh)

    def ST99(self, M, z, dc):
        """
        Peak-background split bias corresponding to ST99 HMF.

        Taken from Sheth & Tormen (1999).
        """
        nu = dc / self.sigmaM(M, z)

        q = self.bias_par.get("q", 0.707)
        p = self.bias_par.get("p", 0.3)

        return 1.0 + (q * nu**2 - 1.0) / dc + (2.0 * p / dc) / (1.0 + (q * nu**2) ** p)

    def SMT01(self, M, z, dc):
        """
        Extended Press-Schechter-derived bias function corresponding to SMT01 HMF

        Taken from Sheth, Mo & Tormen (2001)
        """
        nu = dc / self.sigmaM(M, z)

        a = self.bias_par.get("a", 0.707)
        b = self.bias_par.get("b", 0.5)
        c = self.bias_par.get("c", 0.6)

        sa = a**0.5
        return 1.0 + (
            sa * (a * nu**2.0)
            + sa * b * (a * nu**2.0) ** (1.0 - c)
            - (a * nu**2.0) ** c
            / ((a * nu**2.0) ** c + b * (1.0 - c) * (1.0 - c / 2.0))
        ) / (dc * sa)

    def Seljak04(self, M, z, dc):
        """
        Empirical bias relation from Seljak & Warren (2004), without cosmological dependence.
        """
        M = np.atleast_1d(M)
        z = np.atleast_1d(z)

        M_NL = self.astro.mass_non_linear(z, delta_crit=dc)
        x = (M[:, None] / M_NL[None, :]).to(1).value

        a = self.bias_par.get("a", 0.53)
        b = self.bias_par.get("b", 0.39)
        c = self.bias_par.get("c", 0.45)
        d = self.bias_par.get("d", 0.13)
        e = self.bias_par.get("e", 40.0)
        f = self.bias_par.get("f", 5e-4)
        g = self.bias_par.get("g", 1.5)

        bh = a + b * x**c + d / (e * x + 1.0) + f * x**g
        return np.squeeze(bh)

    def Seljak04_Cosmo(self, M, z, dc):
        """
        Empirical bias relation from Seljak & Warren (2004), with cosmological dependence.
        Doesn't include the running of the spectral index alpha_s.
        """
        M = np.atleast_1d(M)
        z = np.atleast_1d(z)

        M_NL = self.astro.mass_non_linear(z, delta_crit=dc)
        x = (M[:, None] / M_NL[None, :]).to(1).value

        a = self.bias_par.get("a", 0.53)
        b = self.bias_par.get("b", 0.39)
        c = self.bias_par.get("c", 0.45)
        d = self.bias_par.get("d", 0.13)
        e = self.bias_par.get("e", 40.0)
        f = self.bias_par.get("f", 5e-4)
        g = self.bias_par.get("g", 1.5)
        a1 = self.bias_par.get("a1", 0.4)
        a2 = self.bias_par.get("a2", 0.3)
        a3 = self.bias_par.get("a3", 0.8)

        Om0m = self.cosmology.Omegam_of_z(0)
        ns = self.cosmology.input_cosmopars.get(
            "ns", self.cosmology.input_cosmopars["n_s"]
        )
        nrun = self.cosmology.input_cosmopars.get(
            "alpha_s", self.cosmology.input_cosmopars["nrun"]
        )
        s8 = self.cosmology.sigma8_of_z(0)

        bh = (
            a
            + b * x**c
            + d / (e * x + 1.0)
            + f * x**g
            + np.log10(x)
            * (
                a1 * (Om0m - 0.3 + ns - 1.0)
                + a2 * (s8 - 0.9 + self.hubble - 0.7)
                + a3 * nrun
            )
        )
        return np.squeeze(bh)

    def Tinker05(self, M, z, dc):
        """
        Empirical bias, same as SMT01 but modified parameters.
        """
        nu = dc / self.sigmaM(M, z)

        a = self.bias_par.get("a", 0.707)
        b = self.bias_par.get("b", 0.35)
        c = self.bias_par.get("c", 0.8)

        sa = a**0.5
        return 1.0 + (
            sa * (a * nu**2)
            + sa * b * (a * nu**2) ** (1.0 - c)
            - (a * nu**2) ** c / ((a * nu**2) ** c + b * (1.0 - c) * (1.0 - c / 2.0))
        ) / (dc * sa)

    def Mandelbaum05(self, M, z, dc):
        """
        Empirical bias, same as ST99 but changed parameters
        """
        nu = dc / self.sigmaM(M, z)

        q = self.bias_par.get("q", 0.73)
        p = self.bias_par.get("p", 0.15)

        return (
            1.0 + (q * nu**2.0 - 1.0) / dc + (2.0 * p / dc) / (1.0 + (q * nu**2.0) ** p)
        )

    def Manera10(self, M, z, dc):
        """
        Empirical bias, same as ST99 but changed parameters
        """
        nu = dc / self.sigmaM(M, z)

        q = self.bias_par.get("q", 0.709)
        p = self.bias_par.get("p", 0.248)

        return (
            1.0 + (q * nu**2.0 - 1.0) / dc + (2.0 * p / dc) / (1.0 + (q * nu**2.0) ** p)
        )

    def constant(self, M, z, dc):
        """
        Returns a linear constant bias
        """
        nu = dc / self.sigmaM(M, z)

        return self.bias_par["b"] * np.ones_like(nu)
