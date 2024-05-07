# This class defines global variables that will not change through
# the computation of one fisher matrix

import sys, os
import yaml



from copy import copy
from astropy import units as u
import numpy as np

from SSLimPy.cosmology import cosmology

def init(
    settings_dict = dict(),
    camb_yaml_file = None,
    class_yaml_file = None,
    specifications = None,
    cosmopars = dict()
):
    """This class is to handle the configuration for the computation as well as the fiducial parameters. It then gives access to all global variables

    Parameters
    ----------
    settings        : dict
                      A dictionary containing all additional neccessary options for the computation
    specifications  : String
                      A path to the survey specifications. Defaults to the default
    camb_yaml_file  : String
                      A path containing the configuration of the EBS camb. Defaults to the default
    class_yaml_file : String
                      A path containing the configuration of the EBS class. Defaults to the default
    cosmopars       : dict, optional
                      A dictionary containing the fiducial cosmological parameters
    """

    #Set global defaults for the settings dictionary
    global settings
    settings = settings_dict
    settings.setdefault("cosmo_model", "LCDM")
    settings.setdefault("code", "camb")
    settings.setdefault("h-units", False)
    settings.setdefault("kmin", 1.e-2 * u.Mpc**-1)
    settings.setdefault("kmax", 10 * u.Mpc**-1)
    settings.setdefault("nk",100)
    settings.setdefault("k_kind","log")
    settings.setdefault("sigma_scatter",0)
    settings.setdefault("fduty",1)
    settings.setdefault("do_onehalo", False)
    settings.setdefault("do_Jysr", False)
    settings.setdefault("do_RSD", True)
    settings.setdefault("sigma_NL", 7)
    settings.setdefault("nmu", 1000)
    settings.setdefault("FoG_damp", "Lorentzian")
    settings.setdefault("smooth", False)
    settings.setdefault("do_conv_Wkmin", False)
    settings.setdefault("nonlinear", False)
    settings.setdefault("share_delta_neff", True)
    settings.setdefault("LP_rescale_ini_As", 2.1e-9)
    settings.setdefault("LP_rescale_boost", 2)
    settings.setdefault("nonlinear", True)
    settings.setdefault("output", None)

    # Load Boltzmann solver files
    global input_type
    input_type = settings["code"]

    file_location = os.path.dirname(__file__)

    if input_type == "camb":
        global boltzmann_cambpars
        if camb_yaml_file:
            if os.path.exists(camb_yaml_file):
                file_content_camb = yaml.safe_load(open(camb_yaml_file))
            else:
                print("You asked for CAMB but the yaml path you passed did not exist")
                raise ValueError
        else:
            file_content_camb = yaml.safe_load(open(os.path.join(file_location,"../../input/solver_files/camb_default.yaml")))
        boltzmann_cambpars = file_content_camb

    if input_type == "class":
        global boltzmann_classpars
        if class_yaml_file:
            if os.path.exists(class_yaml_file):
                file_content_class = yaml.safe_load(open(class_yaml_file))
            else:
                print("You asked for CLASS but the yaml path you passed did not exist")
                raise ValueError
        else:
            file_content_class = yaml.safe_load(open(os.path.join(file_location,"../../input/solver_files/class_default.yaml")))
        boltzmann_classpars = file_content_class

    # Load Survey specifications from file
    if specifications:
        if os.path.exists(specifications):
            file_content_specs =  yaml.safe_load(open(specifications))
        else:
            print("The specified path to the survey specifications does not exist")
            raise ValueError
    else:
        file_content_specs = yaml.safe_load(open(os.path.join(file_location,"../../input/survey_files/default.yaml")))

    # restore units and fill seperate spec dirs
    global astropars
    astropars = copy(file_content_specs["ASTRO"])

    astropars["nu"] *= u.GHz
    astropars["nuObs"] *= u.GHz
    astropars["Mmin"] *= u.Msun
    astropars["Mmax"] *= u.Msun
    astropars["Lmin"] *= u.Lsun
    astropars["Lmax"] *= u.Lsun

    if astropars["model_name"] == "SchCut" or astropars["model_name"] == "Sch":
        astropars["model_par"]["phistar"] *= u.Lsun**-1*u.Mpc**-3
        astropars["model_par"]["Lstar"] *= u.Lsun
        astropars["model_par"]["Lmin"] *= u.Lsun

    global vidpars
    vidpars = copy(file_content_specs["VID"])

    vidpars["Tmin_VID"] *= u.uK
    vidpars["Tmax_VID"] *= u.uK
    vidpars["fT0_min"] *= u.uK**-1
    vidpars["fT0_max"] *= u.uK**-1
    vidpars["fT_min"] *= u.uK**-1
    vidpars["fT_max"] *= u.uK**-1
    vidpars["sigma_PT_stable"] *= u.uK
    vidpars["nT"]=np.exp(vidpars["lognT"])

    global obspars
    obspars = copy(file_content_specs["OBS"])

    obspars["Tsys_NEFD"] *= u.K
    obspars["beam_FWHM"] *= u.arcmin
    obspars["Delta_nu"] *= u.GHz
    obspars["dnu"] *= u.MHz
    obspars["tobs"] *= u.hr
    obspars["Omega_field"] *= u.deg**2
    obspars["a_FG"] *= u.Mpc**-1

    global fiducialparams
    fiducialparams = cosmopars

    global fiducialcosmo
    fiducialcosmo = cosmology.cosmo_functions(cosmopars,input_type)