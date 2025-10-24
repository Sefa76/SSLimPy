"""
Calculate the higher order bias functions from co-evolution relationships.

For now all functions assume a underlying spherical collapse (ST, SMT, Tinker) b(M,z). Equations are taken from [1802.07622]

The same signatures of the bias functions in ```bias_fitting_functions``` should be kept
"""

import numpy as np

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
        """dummy function
        """
        return np.ones_like(M.value)

    def b1(self, M, z, dc):
        """Sheth, Mo, Torman (2001)
        """
        anu2 = self._alpha * (dc / self.sigmaM(M, z))**2
        b1 = (
            1 / (np.sqrt(self._alpha) * dc)
            * (
                np.sqrt(self._alpha) * (anu2)
                + np.sqrt(self._alpha) * self._b * anu2**(1- self._c)
                - anu2**self._c / (
                    anu2**self._c
                    + self._b * (1 - self._c) * (1 - self._c / 2)
                )
            )
        )
        return 1 + b1

    def b2(self, M, z, dc):
        """Schmidt et al. fitting formula for b_2
        """
        b1 = getattr(
            self, self.halomodel.haloparams["bias_model"],
            self.b1,
        )(M, z, dc)
        b2 = 0.412 - 2.143 * b1 + 0.929 * b1**2 + 0.008 * b1**3
        return b2

    def b3(self, M, z, dc):
        """Schmidt et al. fitting formula for b_3
        """
        b1 = getattr(
            self, self.halomodel.haloparams["bias_model"],
            self.b1,
        )(M, z, dc)
        b3 = -1.028 + 7.646 * b1 - 6.227 * b1**2 + 0.912 * b1**3
        return b3

    ##################
    # Non-Local Bias #
    ##################
    # Obtained from co-evolution of the Lagrangian bias expansion

    def bG2(self, M, z, dc):
        return -2 / 7 * (self.b1(M, z, dc) - 1)

    def bG3(self, M, z, dc):
        return -22 / 63 * (self.b1(M, z, dc) - 1)

    def bDG2(self, M, z, dc):
        return 23 / 42 * (self.b1(M, z, dc) - 1)

    def bdG2(self, M, z, dc):
        return -2 / 7 * 2 * (self.b2(M, z, dc) - 4 / 21 * self.b1(M, z, dc))
