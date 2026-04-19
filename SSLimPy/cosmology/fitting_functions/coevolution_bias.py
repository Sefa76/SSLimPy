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
        self._alpha = self.bias_par.get("SMT_alpha", 0.707)
        self._b = self.bias_par.get("SMTb", 0.5)
        self._c = self.bias_par.get("SMTc", 0.6)

    ##############
    # Local Bias #
    ##############
    # Obtained from Lazeyras, T. et al. (2016)

    def b0(self, M, z, dc):
        """dummy function"""
        return np.ones_like(M.value)

    def b1(self, M, z, dc):
        """Sheth, Mo, Torman (2001) from collosus"""
        nu = dc / self.sigmaM(M, z)
        a = self._alpha
        b = self._b
        c = self._c

        roota = np.sqrt(a)
        anu2 = a * nu**2
        anu2c = anu2**c
        t1 = b * (1.0 - c) * (1.0 - 0.5 * c)
        bias = 1.0 +  1.0 / (roota * dc) * (roota * anu2 + roota * b * anu2**(1.0 - c) - anu2c / (anu2c + t1))
        return bias

    def b2sph_SMT(self, M, z, dc):
        """b2 from extended ps model presented in Sheth, Mo, Torman (2001)
        """
        nu = dc / self.sigmaM(M, z)
        a = self._alpha
        b = self._b
        c = self._c
        

    def b2_fitted(self, M, z, dc):
        """b2 from assuming Lazeyras et al fitting + coevolution
        """
        return self.b2sph_lazeyras(M, z, dc) + 4 / 3 * self.bG2(M, z, dc)

    def b2sph_lazeyras(self, M, z, dc):
        """Lazeyras et al. fitting formula for b_2 - 4/3 b_G2"""
        b1 = getattr(
            self,
            self.halomodel.haloparams["bias_model"],
            self.b1,
        )(M, z, dc)
        b2sph = 0.412 - 2.143 * b1 + 0.929 * b1**2 + 0.008 * b1**3
        return b2sph

    def b2sph_euclid(self, M, z, dc):
        """Fitting formula for b_2 - 4/3 b_G2 from Euclid DR1-JC6"""
        b1 = getattr(
            self,
            self.halomodel.haloparams["bias_model"],
            self.b1,
        )(M, z, dc)
        bias =  -0.015 - 1.58 * b1 + 0.809 * b1**2 + 0.025 * b1**3
        return bias

    def b3(self, M, z, dc):
        """Lazeyras et al. fitting formula for b_3"""
        b1 = getattr(
            self,
            self.halomodel.haloparams["bias_model"],
            self.b1,
        )(M, z, dc)
        b3 = -1.028 + 7.646 * b1 - 6.227 * b1**2 + 0.912 * b1**3
        return b3

    ##################
    # Non-Local Bias #
    ##################
    # Obtained from co-evolution of the Lagrangian bias expansion

    def bG2(self, M, z, dc):
        b1 = getattr(
            self,
            self.halomodel.haloparams["bias_model"],
            self.b1,
        )(M, z, dc)
        return -2 / 7 * (b1 - 1)

    def bG3(self, M, z, dc):
        b1 = getattr(
            self,
            self.halomodel.haloparams["bias_model"],
            self.b1,
        )(M, z, dc)
        return -22 / 63 * (b1 - 1)

    def bDG2(self, M, z, dc):
        b1 = getattr(
            self,
            self.halomodel.haloparams["bias_model"],
            self.b1,
        )(M, z, dc)
        return 23 / 42 * (b1 - 1)

    def bdG2(self, M, z, dc):
        b1 = getattr(
            self,
            self.halomodel.haloparams["bias_model"],
            self.b1,
        )(M, z, dc)
        return -2 / 7 * 2 * (self.b2_fitted(M, z, dc) - 4 / 21 * b1)
