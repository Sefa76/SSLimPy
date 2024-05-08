import sys
import numpy as np
import astropy.units as u
import astropy.constants as c

from copy import deepcopy

sys.path.append("../")
from SSLimPy.interface import config as cfg
from SSLimPy.cosmology import cosmology


class astro_functions:
    def __init__(self, cosmopars=dict(), astropars=dict()):

        self.astroparams = deepcopy(astropars)
        self.set_astrophysics_defaults()

        if cosmopars:
            self.cosmopars = deepcopy(cfg.fiducialcosmoparams)
        else:
            self.cosmopars = deepcopy(cosmopars)

        if self.cosmopars == cfg.fiducialcosmoparams:
            # No need to recompute the cosmology
            self.cosmology = cfg.fiducialcosmo
        else:
            self.cosmology = cosmology.cosmo_functions(cosmopars, cfg.input_type)

        # current Units
        self.hubble = self.cosmology.Hubble(0,True)/  (100 * u.km/u.s/u.Mpc)
        self.Mpch = u.Mpc / self.hubble
        self.Msunh = u.Msun / self.hubble

    def set_astrophysics_defaults(self):
        """
        Fills up default values in the astropars dictionary if the values are not found.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        self.astroparams.setdefault("model_type", "LF")
        self.astroparams.setdefault("model_name", "SchCut")
        self.astroparams.setdefault(
            "model_par",
            {
                "phistar": 9.6e-11 * u.Lsun**-1 * u.Mpc**-3,
                "Lstar": 2.1e6 * u.Lsun,
                "alpha": -1.87,
                "Lmin": 5000 * u.Lsun,
            },
        )
        self.astroparams.setdefault("hmf_model", "ST")
        self.astroparams.setdefault("bias_model", "ST99")
        self.astroparams.setdefault("bias_par", {})
        self.astroparams.setdefault("nu", 115 * u.GHz)
        self.astroparams.setdefault("nuObs", 30 * u.GHz)
        self.astroparams.setdefault("Mmin", 1e9 * u.Msun)
        self.astroparams.setdefault("Mmax", 1e15 * u.Msun)
        self.astroparams.setdefault("nM", 500)
        self.astroparams.setdefault("Lmin", 10 * u.Lsun)
        self.astroparams.setdefault("Lmax", 1e8 * u.Lsun)
        self.astroparams.setdefault("nL", 5000)
        self.astroparams.setdefault("v_of_M", None)
        self.astroparams.setdefault("line_incli", True)

    def sigmaM(self, M, z, tracer = "clustering"):
        '''
        Mass (or cdm+b) variance at target redshift
        '''
        #Get R(M) and P(k)
        rho_crit = 2.77536627e11*(self.Msunh*self.Mpch**-3).to(u.Msun*u.Mpc**-3) #Msun/Mpc^3
        rhoM = rho_crit*self.cosmology.Omega(0,tracer)
        R = (3.0*M/(4.0*np.pi*rhoM))**(1.0/3.0)

        return self.cosmology.sigmaR_of_z(z,R,tracer)
    
    def recap_astro(self):
        print("Astronomical Parameters:")
        for key in self.astroparams:
            print("   " + key + ": {}".format(self.astroparams[key]))