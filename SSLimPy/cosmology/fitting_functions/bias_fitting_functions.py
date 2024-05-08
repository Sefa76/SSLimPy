"""
Calculate the halo bias depending on the mass for different fitting functions

All functions return b(M)

Bias model parameter values are given by a dictionary called bias_par.  Each
function takes a value of dc, nu
"""

import numpy as np


def Tinker10(self, dc, nu):
    # Parameters of bias fit
    if len(self.bias_par.keys()) == 0:
        y = np.log10(200.0)
        A = 1.0 + 0.24 * y * np.exp(-((4.0 / y) ** 4.0))
        a = 0.44 * y - 0.88
        B = 0.183
        b = 1.5
        C = 0.019 + 0.107 * y + 0.19 * np.exp(-((4.0 / y) ** 4.0))
        c = 2.4
    else:
        y = self.bias_par["y"]
        B = self.bias_par["B"]
        b = self.bias_par["b"]
        c = self.bias_par["c"]
        A = 1.0 + 0.24 * y * np.exp(-((4.0 / y) ** 4.0))
        C = 0.019 + 0.107 * y + 0.19 * np.exp(-((4.0 / y) ** 4.0))
        a = 0.44 * y - 0.88

    return 1.0 - A * nu**a / (nu**a + dc**a) + B * nu**b + C * nu**c


def Mo96(self, dc, nu):
    """
    Peak-background split bias correspdonding to PS HMF.

    Taken from Mo and White (1996)
    """
    return 1.0 + (nu**2.0 - 1.0) / dc


def Jing98(self, dc, nu):
    """
    Empirical bias of Jing (1998): http://adsabs.harvard.edu/abs/1998ApJ...503L...9J
    """
    Mh = (self.M.to(self.Msunh)).value
    mass_non_linear = (np.argmin((self.sigmaM - 1) ** 2.0).to(self.Msunh)).value
    if self.cosmo_code == "camb":
        ns = self.cosmo_input_camb["ns"]
    else:
        ns = self.cosmo.n_s()
    nu_star = (Mh / mass_non_linear) ** (ns + 3.0) / 6.0
    if len(self.bias_par.keys()) == 0:
        a = 0.5
        b = 0.06
        c = 0.02
    else:
        a = self.bias_par["a"]
        b = self.bias_par["b"]
        c = self.bias_par["c"]
    return (a / nu_star**4.0 + 1.0) ** (b - c * ns) * (1.0 + (nu_star**2.0 - 1.0) / dc)


def ST99(self, dc, nu):
    """
    Peak-background split bias corresponding to ST99 HMF.

    Taken from Sheth & Tormen (1999).
    """
    if len(self.bias_par.keys()) == 0:
        q = 0.707
        p = 0.3
    else:
        q = self.bias_par["q"]
        p = self.bias_par["p"]
    return 1.0 + (q * nu**2 - 1.0) / dc + (2.0 * p / dc) / (1.0 + (q * nu**2) ** p)


def SMT01(self, dc, nu):
    """
    Extended Press-Schechter-derived bias function corresponding to SMT01 HMF

    Taken from Sheth, Mo & Tormen (2001)
    """
    if len(self.bias_par.keys()) == 0:
        a = 0.707
        b = 0.5
        c = 0.6
    else:
        a = self.bias_par["a"]
        b = self.bias_par["b"]
        c = self.bias_par["c"]
    sa = a**0.5
    return 1.0 + (
        sa * (a * nu**2.0)
        + sa * b * (a * nu**2.0) ** (1.0 - c)
        - (a * nu**2.0) ** c / ((a * nu**2.0) ** c + b * (1.0 - c) * (1.0 - c / 2.0))
    ) / (dc * sa)


def Seljak04(self, dc, nu):
    """
    Empirical bias relation from Seljak & Warren (2004), without cosmological dependence.
    """
    mass_non_linear = (np.argmin((self.sigmaM - dc) ** 2.0).to(self.Msunh)).value
    Mh = (self.M.to(self.Msunh)).value
    x = Mh / self.mass_non_linear
    if len(self.bias_par.keys()) == 0:
        a = 0.53
        b = 0.39
        c = 0.45
        d = 0.13
        e = 40.0
        f = 5e-4
        g = 1.5
    else:
        a = self.bias_par["a"]
        b = self.bias_par["b"]
        c = self.bias_par["c"]
        d = self.bias_par["d"]
        e = self.bias_par["e"]
        f = self.bias_par["f"]
        g = self.bias_par["g"]
    return a + b * x**c + d / (e * x + 1.0) + f * x**g


def Seljak04_Cosmo(self, dc, nu):
    """
    Empirical bias relation from Seljak & Warren (2004), with cosmological dependence.
    Doesn't include the running of the spectral index alpha_s.
    """
    mass_non_linear = (np.argmin((self.sigmaM - dc) ** 2.0).to(self.Msunh)).value
    Mh = (self.M.to(self.Msunh)).value
    x = Mh / self.mass_non_linear
    if len(self.bias_par.keys()) == 0:
        a = 0.53
        b = 0.39
        c = 0.45
        d = 0.13
        e = 40.0
        f = 5e-4
        g = 1.5
        a1 = 0.4
        a2 = 0.3
        a3 = 0.8
    else:
        a = self.bias_par["a"]
        b = self.bias_par["b"]
        c = self.bias_par["c"]
        d = self.bias_par["d"]
        e = self.bias_par["e"]
        f = self.bias_par["f"]
        g = self.bias_par["g"]
        a1 = self.bias_par["a1"]
        a2 = self.bias_par["a2"]
        a3 = self.bias_par["a3"]
    if self.cosmo_code == "camb":
        Om0m = self.camb_pars.omegam
        ns = self.cosmo_input_camb["ns"]
        s8 = self.cosmo.get_sigma8_0()
        nrun = self.cosmo_input_camb["nrun"]
    else:
        Om0m = self.cosmo.Omega0_m()
        ns = self.cosmo.n_s()
        s8 = self.cosmo.sigma8()
        try:
            nrun = self.cosmo_input_class["alpha_s"]
        except:
            nrun = 0.0
    return (
        a
        + b * x**c
        + d / (e * x + 1.0)
        + f * x**g
        + np.log10(x)
        * (
            a1 * (Om0m - 0.3 + ns - 1.0)
            + a2 * (self.s8 - 0.9 + self.hubble - 0.7)
            + a3 * nrun
        )
    )


def Tinker05(self, dc, nu):
    """
    Empirical bias, same as SMT01 but modified parameters.
    """
    if len(self.bias_par.keys()) == 0:
        a = 0.707
        b = 0.35
        c = 0.8
    else:
        a = self.bias_par["a"]
        b = self.bias_par["b"]
        c = self.bias_par["c"]
    sa = a**0.5
    return 1.0 + (
        sa * (a * nu**2)
        + sa * b * (a * nu**2) ** (1.0 - c)
        - (a * nu**2) ** c / ((a * nu**2) ** c + b * (1.0 - c) * (1.0 - c / 2.0))
    ) / (dc * sa)


def Mandelbaum05(self, dc, nu):
    """
    Empirical bias, same as ST99 but changed parameters
    """
    if len(self.bias_par.keys()) == 0:
        q = 0.73
        p = 0.15
    else:
        q = self.bias_par["q"]
        p = self.bias_par["p"]
    return 1.0 + (q * nu**2.0 - 1.0) / dc + (2.0 * p / dc) / (1.0 + (q * nu**2.0) ** p)


def Manera10(self, dc, nu):
    """
    Empirical bias, same as ST99 but changed parameters
    """
    if len(self.bias_par.keys()) == 0:
        q = 0.709
        p = 0.248
    else:
        q = self.bias_par["q"]
        p = self.bias_par["p"]
    return 1.0 + (q * nu**2.0 - 1.0) / dc + (2.0 * p / dc) / (1.0 + (q * nu**2.0) ** p)


def constant(self, dc, nu):
    """
    Returns a linear constant bias
    """
    return self.bias_par["b"] * np.ones(len(nu))
