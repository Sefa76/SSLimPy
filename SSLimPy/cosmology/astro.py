import sys
import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
import astropy.constants as c

from copy import deepcopy

sys.path.append("../")
from SSLimPy.interface import config as cfg
from SSLimPy.cosmology import cosmology
from SSLimPy.cosmology.fitting_functions import bias_fitting_functions as bf
from SSLimPy.cosmology.fitting_functions import halo_mass_functions as HMF
from SSLimPy.cosmology.fitting_functions import luminosity_functions as lf
from SSLimPy.cosmology.fitting_functions import mass_luminosity as ml

class astro_functions:
    def __init__(self, cosmopars=dict(), astropars=dict()):

        self.astroparams = deepcopy(astropars)
        self.set_astrophysics_defaults()

        ### TEXT VOMIT ###
        if cfg.settings["verbosity"]>1:
            self.recap_astro()
        ##################

        if cosmopars:
            self.cosmopars = deepcopy(cfg.fiducialcosmoparams)
        else:
            self.cosmopars = deepcopy(cosmopars)

        if self.cosmopars == cfg.fiducialcosmoparams:
            # No need to recompute the cosmology
            self.cosmology = cfg.fiducialcosmo
        else:
            self.cosmology = cosmology.cosmo_functions(cosmopars, cfg.input_type)

        # Check passed models
        # self.check_model()
        # self.check_halo_mass_function_model()
        self.init_bias_model()

        # current Units
        self.hubble = self.cosmology.Hubble(0,True)/  (100 * u.km/u.s/u.Mpc)
        self.Mpch = u.Mpc / self.hubble
        self.Msunh = u.Msun / self.hubble
        self.rho_crit = 2.77536627e11*(self.Msunh*self.Mpch**-3).to(u.Msun*u.Mpc**-3) #Msun/Mpc^3

        self.M = np.geomspace(self.astroparams["Mmin"],self.astroparams["Mmax"],self.astroparams["nM"])
        self.L = np.geomspace(self.astroparams["Lmin"],self.astroparams["Lmax"],self.astroparams["nL"])
        self.z = cfg.settings["z_output"]

        self.sigmaM, self.dsigmaM_dM = self.compute_sigmaM_funcs(self.M,self.z)

    def sigmaM_of_z(self, M, z, tracer = "clustering"):
        '''
        Mass (or cdm+b) variance at target redshift
        '''
        rhoM = self.rho_crit*self.cosmology.Omega(0,tracer)
        R = (3.0*M/(4.0*np.pi*rhoM))**(1.0/3.0)

        return self.cosmology.sigmaR_of_z(z,R,tracer)
    
    def compute_sigmaM_funcs(self, M, z, tracer="clustering"):
        sigmaM = self.sigmaM_of_z(M,z,tracer=tracer)
        sigmaMpe = self.sigmaM_of_z(M*(1+1e-2),z,tracer=tracer)
        sigmaMme = self.sigmaM_of_z(M*(1-1e-2),z,tracer=tracer)
        dM = 2e-2*M

        dsigmaM_dM = (sigmaMpe + sigmaMme - 2* sigmaM) / dM

        return sigmaM, dsigmaM_dM


    def mass_non_linear(self, z, delta_crit =1.686, tracer = "clustering"):
        '''
        Get (roughly) the mass corresponding to the nonlinear scale in units of Msun h
        '''
        sigmaM_z = self.sigmaM_of_z(self.M,z,tracer=tracer)
        mass_non_linear = self.M[np.argmin(np.power(sigmaM_z - delta_crit,2),axis=0)]

        return mass_non_linear.to(self.Msunh)

    ##################
    # Helper Functions
    ##################
    # To be Outsorced

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

    def check_model(self):
        '''
        Check if model given by model_name exists in the given model_type
        '''
        model_type = self.astroparams["model_type"]
        model_name = self.astroparams["model_name"]

        if model_type=='ML' and not hasattr(ml,model_name):
            if hasattr(lf,model_name):
                raise ValueError(model_name+" not found in mass_luminosity.py."+
                        " Set model_type='LF' to use "+model_name)
            else:
                raise ValueError(model_name+
                        " not found in mass_luminosity.py")
        elif model_type=='LF' and not hasattr(lf,model_name):
            if hasattr(ml,model_name):
                raise ValueError(model_name+
                        " not found in luminosity_functions.py."+
                        " Set model_type='ML' to use "+model_name)
            else:
                raise ValueError(model_name+
                        " not found in luminosity_functions.py")
                
    def init_bias_model(self):
        '''
        Check if model given by bias_model exists in the given model_type
        '''
        bias_name = self.astroparams["bias_model"]
        self.bias_functions = bf.bias_fittinig_functions(self)
        if not hasattr(self.bias_functions,bias_name):
            raise ValueError(bias_name+
                        " not found in bias_fitting_functions.py")
                        
    def check_halo_mass_function_model(self):
        '''
        Check if model given by hmf_model exists in the given model_type
        '''
        hmf_model = self.astroparams["hmf_model"]

        if not hasattr(HMF,hmf_model):
            raise ValueError(hmf_model+
                        " not found in halo_mass_functions.py")

    def recap_astro(self):
        print("Astronomical Parameters:")
        for key in self.astroparams:
            print("   " + key + ": {}".format(self.astroparams[key]))