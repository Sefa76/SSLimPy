"""
Calculate the higher order bias functions from co-evolution relationships.

For now all functions assume a underlying spherical collapse (ST, SMT, Tinker) b(M,z). Equations are taken from [1802.07622]

The same signatures of the bias functions in ```bias_fitting_functions``` should be kept
"""

import numpy as np
from warnings import warn
from astropy import units as u

from .bias_fitting_functions import bias_fitting_functions


class coevolution_bias(bias_fitting_functions):

    def __init__(self, halomodel):
        super().__init__(halomodel)
        self._A = self.bias_par.get("HMF_A", 0.322)
        self._p = self.bias_par.get("HMF_p", 0.3)
        self._alpha = self.bias_par.get("HMF_alpha", 0.707)

    ##############
    # Local Bias #
    ##############
    # Peak--Background Split
    def b0(self, M, z, dc):
        """dummy function"""
        return np.ones_like(M.value)

    def b1_HMF(self, M, z, dc):
        """Sheth, Mo, Torman (2001) from collosus"""
        nu = dc / self.sigmaM(M, z)
        alpha = self._alpha
        p = self._p
        x = alpha * nu**2

        eps1 = (x - 1) / dc
        E1 = (2 * p / dc) / (1 + x**p)

        bias = 1 + eps1 + E1
        return bias

    def b2sph_HMF(self, M, z, dc):
        """b_2 - 4/3 b_G2 from extended ps model presented in Sheth, Mo, Torman (2001)"""
        nu = dc / self.sigmaM(M, z)
        alpha = self._alpha
        p = self._p
        x = alpha * nu**2

        eps1 = (x - 1) / dc
        E1 = (2 * p / dc) / (1 + x**p)
        eps2 = x / dc**2 * (x - 3)
        E2 = ((1 + 2 * p) / dc + 2 * eps1) * E1

        a2 = -17 / 21  # I know this number...
        bias = 2 * (1 + a2) * (eps1 + E1) + E2 + eps2
        return bias

    def b3_HMF(self, M, z, dc):
        """b3 from extended ps model presented in Sheth, Mo, Torman (2001)"""
        nu = dc / self.sigmaM(M, z)
        alpha = self._alpha
        p = self._p
        x = alpha * nu**2

        eps1 = (x - 1) / dc
        E1 = (2 * p / dc) / (1 + x**p)
        eps2 = x / dc**2 * (x - 3)
        E2 = ((1 + 2 * p) / dc + 2 * eps1) * E1
        eps3 = 2 * x / dc**3 * (x**2 - 6 * x + 3)
        E3 = ((4 * (p**2 - 1) + 6 * p * x) / dc**2 + 3 * eps1**2) * E1

        a2 = -17 / 21  # I know this number...
        a3 = 341 / 567
        bias = 6 * (a2 + a3) * (eps1 + E1) + 3 * (1 + 2 * a2) * (eps2 + E2) + eps3 + E3
        return bias

    # Obtained from Lazeyras, T. et al. (2016)
    def b2sph_lazeyras(self, M, z, dc):
        """Lazeyras et al. fitting formula for b_2 - 4/3 b_G2"""
        b1 = getattr(
            self,
            self.halomodel.haloparams["bias_model"],
            self.b1_HMF,
        )(M, z, dc)
        b2sph = 0.412 - 2.143 * b1 + 0.929 * b1**2 + 0.008 * b1**3
        return b2sph

    def b3_lazeyras(self, M, z, dc):
        """Lazeyras et al. fitting formula for b_3"""
        b1 = getattr(
            self,
            self.halomodel.haloparams["bias_model"],
            self.b1_HMF,
        )(M, z, dc)
        b3 = -1.028 + 7.646 * b1 - 6.227 * b1**2 + 0.912 * b1**3
        return b3

    # def b2sph_euclid(self, M, z, dc):
    #     """Fitting formula for GALAXIES b_2 - 4/3 b_G2 from Euclid DR1-JC6"""
    #     b1 = getattr(
    #         self,
    #         self.halomodel.haloparams["bias_model"],
    #         self.b1_HMF,
    #     )(M, z, dc)
    #     bias = -0.015 - 1.58 * b1 + 0.809 * b1**2 + 0.025 * b1**3
    #     return bias

    ##################
    # Non-Local Bias #
    ##################
    # Obtained from co-evolution of the Lagrangian bias expansion

    def bG2(self, M, z, dc):
        b1 = getattr(
            self,
            self.halomodel.haloparams["bias_model"],
            self.b1_HMF,
        )(M, z, dc)
        return -2 / 7 * (b1 - 1)

    def bG3(self, M, z, dc):
        b1 = getattr(
            self,
            self.halomodel.haloparams["bias_model"],
            self.b1_HMF,
        )(M, z, dc)
        return -22 / 63 * (b1 - 1)

    def bDG2(self, M, z, dc):
        b1 = getattr(
            self,
            self.halomodel.haloparams["bias_model"],
            self.b1_HMF,
        )(M, z, dc)
        return 23 / 42 * (b1 - 1)

    def bdG2(self, M, z, dc):
        b1 = getattr(
            self,
            self.halomodel.haloparams["bias_model"],
            self.b1_HMF,
        )(M, z, dc)
        return -2 / 7 * 2 * (self.b2(M, z, dc) - 4 / 21 * b1)

    ######################
    # Wrapping functions #
    ######################

    def b2sph(self, M, z, dc):
        if self.halomodel.haloparams["nonlinear_bias"] == "fitted":
            return self.b2sph_lazeyras(M, z, dc)
        else:
            return self.b2sph_HMF(M, z, dc)

    def b2(self, M, z, dc):
        b2sph = self.b2sph(M, z, dc)
        bG2 = self.bG2(M, z, dc)
        return b2sph + 4 / 3 * bG2

    def b3(self, M, z, dc):
        if self.halomodel.haloparams["nonlinear_bias"] == "fitted":
            return self.b3_lazeyras(M, z, dc)
        else:
            return self.b3_HMF(M, z, dc)
