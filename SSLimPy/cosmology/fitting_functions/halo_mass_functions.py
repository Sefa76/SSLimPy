"""
Calculate the halo mass function as function of mass for different fitting functions

All functions return dn/dM(M)

Each function takes: self, Mvec, rhoM, z

Takes inspiration from pylians
"""

import numpy as np
from scipy.interpolate import interp1d


class halo_mass_functions:
    def __init__(self, astro):
        self.astro = astro

        # Units and Redshifts
        self.Mpch = astro.Mpch
        self.Msunh = astro.Msunh

        # SigmaM functions
        self.sigmaM = astro.sigmaM
        self.dsigmaM_dM = astro.dsigmaM_dM

    def ST(self, Mvec, rhoM, z):
        """
        Sheth-Tormen halo mass function
        """
        sigmaM = self.sigmaM(Mvec, z)
        dsigmaM_dM = self.dsigmaM_dM(Mvec, z).to(self.Msunh**-1)

        deltac = 1.686
        nu = (deltac / sigmaM) ** 2.0
        nup = 0.707 * nu

        dndM = (
            -2
            * (rhoM / Mvec)[:,None]
            * dsigmaM_dM
            / sigmaM
            * 0.3222
            * (1.0 + 1.0 / nup**0.3)
            * np.sqrt(0.5 * nup)
            * np.exp(-0.5 * nup)
            / np.sqrt(np.pi)
        )

        return np.squeeze(dndM)

    def Tinker(self, Mvec, rhoM, z):
        """
        Tinker et al 2008 halo mass function for delta=200
        """
        delta = 200
        alpha = 10 ** (-((0.75 / np.log10(delta / 75.0)) ** 1.2))

        sigmaM = self.sigmaM(Mvec, z)
        dsigmaM_dM = self.dsigmaM_dM(Mvec, z).to(self.Msunh**-1)

        z_array = np.atleast_1d(z)[None, :]

        # this is for R200_critical with OmegaM=0.2708
        D = 200.0
        A = 0.186 * (1.0 + z_array) ** (-0.14)
        a = 1.47 * (1.0 + z_array) ** (-0.06)
        b = 2.57 * (1.0 + z_array) ** (-alpha)
        c = 1.19

        fs = A * ((b / sigmaM) ** (a) + 1.0) * np.exp(-c / sigmaM**2)

        dndM = -(rhoM / Mvec)[:,None] * dsigmaM_dM.to(self.Msunh**-1) * fs / sigmaM

        return np.squeeze(dndM)

    def Crocce(self, Mvec, rhoM, z):
        """
        Crocce et al. halo mass function
        """
        z_array = np.atleast_1d(z)[None, :]

        sigmaM = self.sigmaM(Mvec, z)
        dsigmaM_dM = self.dsigmaM_dM(Mvec, z).to(self.Msunh**-1)

        A = 0.58 * (1.0 + z_array) ** (-0.13)
        a = 1.37 * (1.0 + z_array) ** (-0.15)
        b = 0.3 * (1.0 + z_array) ** (-0.084)
        c = 1.036 * (1.0 + z_array) ** (-0.024)

        fs = A * (sigmaM ** (-a) + b) * np.exp(-c / sigmaM**2)
        dndM = -(rhoM / Mvec)[:,None] * dsigmaM_dM.to(self.Msunh**-1) * fs / sigmaM

        return np.squeeze(dndM)

    def Jenkins(self, Mvec, rhoM, z):
        """
        Jenkins et al. halo mass function
        """
        A = 0.315
        b = 0.61
        c = 3.8

        sigmaM = self.sigmaM(Mvec, z)
        dsigmaM_dM = self.dsigmaM_dM(Mvec, z).to(self.Msunh**-1)

        fs = A * np.exp(-np.absolute(np.log(1.0 / sigmaM) + b) ** c)

        dndM = -(rhoM / Mvec)[:,None] * dsigmaM_dM.to(self.Msunh**-1) * fs / sigmaM

        return dndM

    def Warren(self, Mvec, rhoM, z):
        """
        Warren et al. halo mass function
        """
        A = 0.7234
        a = 1.625
        b = 0.2538
        c = 1.1982

        sigmaM = self.sigmaM(Mvec, z)
        dsigmaM_dM = self.dsigmaM_dM(Mvec, z).to(self.Msunh**-1)

        fs = A * (sigmaM ** (-a) + b) * np.exp(-c / sigmaM**2)
        dndM = -(rhoM / Mvec)[:,None] * dsigmaM_dM.to(self.Msunh**-1) * fs / sigmaM

        return dndM

    def Watson(self, Mvec, rhoM, z):
        """
        Watson et al. halo mass function for delta=200. Can be changed
        """
        delta = 200.0
        OmegaM = (
            rhoM
            / 2.77536627e11
            * (self.Msunh * self.Mpch**-3).to(self.Msunh * self.Mpch**-3)
        )

        A = 0.194
        a = 1.805
        b = 2.267
        c = 1.287

        sigmaM = self.sigmaM(Mvec, z)
        dsigmaM_dM = self.dsigmaM_dM(Mvec, z).to(self.Msunh**-1)

        factor = (
            np.exp(0.023 * (delta / 178.0 - 1.0))
            * (delta / 178.0) ** (-0.456 * OmegaM - 0.139)
            * np.exp(0.072 * (1 - delta / 178.0) / sigmaM**2.130)
        )
        fs = A * (sigmaM ** (-a) + b) * np.exp(-c / sigmaM**2) * factor

        dndM = -(rhoM / Mvec)[:,None] * dsigmaM_dM.to(self.Msunh**-1) * fs / sigmaM

        return dndM

    def Watson_FOF(self, Mvec, rhoM, z):
        """
        Watson et al. halo mass function using FOF
        """
        A = 0.282
        a = 2.163
        b = 1.406
        c = 1.210

        sigmaM = self.sigmaM(Mvec, z)
        dsigmaM_dM = self.dsigmaM_dM(Mvec, z).to(self.Msunh**-1)

        fs = A * ((b / sigmaM) ** a + 1.0) * np.exp(-c / sigmaM**2)

        dndM = -(rhoM / Mvec)[:,None] * dsigmaM_dM.to(self.Msunh**-1) * fs / sigmaM

        return dndM

    def Angulo(self, Mvec, rhoM, z):
        """
        Angulo et al. halo mass function
        """
        sigmaM = self.sigmaM(Mvec, z)
        dsigmaM_dM = self.dsigmaM_dM(Mvec, z).to(self.Msunh**-1)

        fs = 0.265 * (1.675 / sigmaM + 1.0) ** 1.9 * np.exp(-1.4 / sigmaM**2)

        dndM = -(rhoM / Mvec)[:,None] * dsigmaM_dM.to(self.Msunh**-1) * fs / sigmaM

        return dndM
