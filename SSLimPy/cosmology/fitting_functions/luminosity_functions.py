"""
Calculate line luminosity functions under different parameterizations

All functions take a vector of luminosities in L_sun and return dn/dL in
(Mpc/h)^-3 L_sun^-1

"""

import numpy as np
import astropy.units as u
from copy import deepcopy

class luminosity_functions:
    def __init__(self,astro):
        self.astro = astro
        self.astroparams = deepcopy(astro.astroparams)
        self.model_par = self.astroparams["model_par"]

    def Sch(self, Lvec):
        """
        Schechter function
        dn/dL = phistar * (L/Lstar)^alpha * exp(-L/Lstar)

        Luminosity function parameters:
        phistar   Overall amplitude in (Mpc/h)^-3 L_sun^-1
        Lstar     High-luminosity cutoff location in L_sun
        alpha     Power law slope
        Lmin      Minimum luminosity (hard cutoff)

        >>> LFparams = {'phistar':8.7e-11*u.Lsun**-1*u.Mpc**-3,\
                        'Lstar':2.1e6*u.Lsun,'alpha':-1.87,'Lmin':500*u.Lsun}
        >>> Lvec = np.array([1.e2,1.e4,1.e7]) * u.Lsun
        >>> print Sch(Lvec,LFparams)
        [  0.000...e+00   1.905...e-06   4.017...e-14] 1 / (Mpc3 solLum)
        """

        phistar = self.model_par["phistar"]
        Lstar = self.model_par["Lstar"]
        alpha = self.model_par["alpha"]
        Lmin = self.model_par["Lmin"]

        dndL = phistar * ((Lvec / Lstar).decompose()) ** alpha * np.exp(-Lvec / Lstar)

        if any(Lvec < Lmin):
            dndL[Lvec < Lmin] = 0.0 * dndL.unit

        return dndL.to(u.Mpc**-3 * u.Lsun**-1)


    def SchCut(self, Lvec):
        """
        Schechter function with exponential low-luminosity cutoff
        dn/dL = phistar * (L/Lstar)^alpha * exp(-L/Lstar-Lmin/L)

        Luminosity function parameters:
        phistar   Overall amplitude in (Mpc/h)^-3 L_sun^-1
        Lstar     High-luminosity cutoff location in L_sun
        alpha     Power law slope
        Lmin      Low-lumiosity cutoff location in L_sun

        >>> LFparams = {'phistar':8.7e-11*u.Lsun**-1*u.Mpc**-3,\
                        'Lstar':2.1e6*u.Lsun,'alpha':-1.87,'Lmin':500.*u.Lsun}
        >>> Lvec = np.array([1.e2,1.e4,1.e7]) * u.Lsun
        >>> print SchCut(Lvec,LFparams)
        [  7.088...e-05   1.812...e-06   4.017...e-14] 1 / (Mpc3 solLum)
        """

        phistar = self.model_par["phistar"]
        Lstar = self.model_par["Lstar"]
        alpha = self.model_par["alpha"]
        Lmin = self.model_par["Lmin"]

        dndL = phistar * (Lvec / Lstar) ** alpha * np.exp(-Lvec / Lstar - Lmin / Lvec)

        return dndL.to(u.Mpc**-3 * u.Lsun**-1)
