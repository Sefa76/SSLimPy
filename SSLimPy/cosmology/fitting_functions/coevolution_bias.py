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
        self.p = self.bias_par.get("HO_p", 0.3)
        self.alpha = self.bias_par.get("HO_alpha", 0.707)
        self.A = self.bias_par.get("HO_A", 0.3222)

    def set_model(self, p, alpha, A=None):
        self.p = p
        self.alpha = alpha

        if A is not None:
            self.A = A
        else:
            Xi = np.geomspace(1e-12, 1e3, num=2000)  # Eh... its good enough
            F = self.unnorm_collapsefunction(Xi)
            self.A = 1 / np.trapz(F, np.log(Xi))

    ###########################
    # Base Halo Mass Function #
    ###########################
    # Underlying halo mass function to compute b1, b2, b3 from. Assumes spherical collapse

    def unnorm_collapsefunction(self, Xi):
        """Universal function for the collapsed matter appearing in spherical collapse models
        This function should used to normalize the actual function (i.E find the value for A)
        Xi is nu^2 from the rest of this function package.
        """
        F = (
            (1 + 1 / (self.alpha * Xi) ** self.p)
            * np.sqrt(self.alpha * Xi / (2 * np.pi))
            * np.exp(-self.alpha * Xi / 2)
        )
        return F

    def sc_hmf(self, M, z, dc):
        M = np.atleast_1d(M)
        z = np.atleast_1d(z)

        nu = np.reshape(dc / self.sigmaM(M, z), (*M.shape, *z.shape))

        sigmaM = np.reshape(self.sigmaM(M, z), (*M.shape, *z.shape))
        dsigmaM_dM = np.reshape(
            self.dsigmaM_dM(M, z).to(self.halomodel.Msunh**-1), (*M.shape, *z.shape)
        )
        dlogsigmaM_dM = dsigmaM_dM / sigmaM

        rho_over_M = self.halomodel.rho_tracer / M[:, None]

        dndM = (
            -2
            * self.A
            * rho_over_M
            * dlogsigmaM_dM
            * self.unnorm_collapsefunction(nu**2)
        )
        return dndM.to(u.Msun**-1 * u.Mpc**-3)

    ##############
    # Local Bias #
    ##############
    # Obtained from completeness relations

    def b0(self, M, z, dc):
        """dummy function
        """
        return np.ones_like(M.value)

    def b1(self, M, z, dc):
        nu = dc / self.sigmaM(M, z)

        eps1 = (self.alpha * nu**2 - 1) / dc
        E1 = 2 * self.p / dc * 1 / (1 + (self.alpha * nu**2) ** self.p)
        return 1 + eps1 + E1

    def b2(self, M, z, dc):
        nu = dc / self.sigmaM(M, z)

        eps1 = (self.alpha * nu**2 - 1) / dc
        E1 = 2 * self.p / dc * 1 / (1 + (self.alpha * nu**2) ** self.p)
        eps2 = self.alpha * nu**2 / dc**2 * (self.alpha * nu**2 - 3)
        E2 = ((1 + 2 * self.p) / dc + 2 * eps1) * E1
        return 2 * (1 - 17 / 21) * (eps1 + E1) + eps2 + E2

    def b3(self, M, z, dc):
        nu = dc / self.sigmaM(M, z)

        eps1 = (self.alpha * nu**2 - 1) / dc
        E1 = 2 * self.p / dc * 1 / (1 + (self.alpha * nu**2) ** self.p)
        eps2 = self.alpha * nu**2 / dc**2 * (self.alpha * nu**2 - 3)
        E2 = ((1 + 2 * self.p) / dc + 2 * eps1) * E1
        eps3 = (
            self.alpha
            * nu**2
            / dc**3
            * (self.alpha**2 * nu**4 - 6 * self.alpha * nu**2 + 3)
        )
        E3 = (
            (4 * (self.p**2 - 1) + 6 * self.p * self.alpha * nu**2) / dc**2
            + 3 * eps1**2
        ) * E1
        return (
            6 * (-17 / 21 + 341 / 567) * (eps1 + E1)
            + 3 * (1 + 2 * 341 / 567) * (eps2 + E2)
            + eps3
            + E3
        )

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
