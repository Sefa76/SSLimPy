# This class defines global variables that will not change through
# the computation of one fisher matrix

import sys, os
import yaml



from copy import copy
from astropy import units as u
import numpy as np

from SSLimPy.cosmology import cosmology
from SSLimPy.cosmology import astro

def init(
    settings_dict = dict(),
    camb_yaml_file = None,
    class_yaml_file = None,
    specifications = None,
    cosmopars = dict(),
    astropars = dict(),
    BAOpars = dict(),
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

    # Cosmology settings
    settings.setdefault("nonlinearMatpow", True)
    settings.setdefault("share_delta_neff", True)
    settings.setdefault("LP_rescale_ini_As", 2.1e-9)
    settings.setdefault("LP_rescale_boost", 2)
    settings.setdefault("cosmo_model", "LCDM")
    settings.setdefault("code", "camb")
    settings.setdefault("h-units", False)
    settings.setdefault("do_pheno_ncdm",False)
    settings.setdefault("astro_tracer", "clustering")

    # Savgol numerics
    settings.setdefault('savgol_window', 101)
    settings.setdefault('savgol_polyorder', 3)
    settings.setdefault('savgol_width', 1.358528901113328)
    settings.setdefault('savgol_internalsamples', 800)
    settings.setdefault('savgol_internalkmin', 0.001)

    # Pk numerics
    settings.setdefault("kmin", 1.e-3 * u.Mpc**-1)
    settings.setdefault("kmax", 10 * u.Mpc**-1)
    settings.setdefault("nk",200)
    settings.setdefault("k_kind","log")
    settings.setdefault("nmu", 128)
    settings.setdefault("downsample_conv_q",1)
    settings.setdefault("downsample_conv_muq",8)

    # Pk specifications
    settings.setdefault("sigma_scatter",0)
    settings.setdefault("fduty",1)
    settings.setdefault("do_Jysr", False)
    settings.setdefault("fix_cosmo_nl_terms",True)

    # Pk contributions
    settings.setdefault("QNLpowerspectrum", True)
    settings.setdefault("TracerPowerSpectrum", "matter")
    settings.setdefault("do_RSD", True)
    settings.setdefault("nonlinearRSD", True)
    settings.setdefault("FoG_damp", "Lorentzian")
    settings.setdefault("halo_model_PS", False)
    settings.setdefault("Smooth_resolution", True)
    settings.setdefault("Smooth_window", False)

    # Output settings
    settings.setdefault("verbosity",1)
    settings.setdefault("output", [])

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

    global vidpars
    vidpars = copy(file_content_specs["VID"])

    vidpars["Tmin_VID"] *= u.uK
    vidpars["Tmax_VID"] *= u.uK
    vidpars["fT0_min"] *= u.uK**-1
    vidpars["fT0_max"] *= u.uK**-1
    vidpars["fT_min"] *= u.uK**-1
    vidpars["fT_max"] *= u.uK**-1
    vidpars["sigma_PT_stable"] *= u.uK
    vidpars["nT"]=int(np.power(2,vidpars["lognT"]))

    global obspars
    obspars = copy(file_content_specs["OBS"])

    obspars["Tsys_NEFD"] *= u.K
    obspars["beam_FWHM"] *= u.arcmin
    obspars["Delta_nu"] *= u.GHz
    obspars["dnu"] *= u.MHz
    obspars["tobs"] *= u.hr
    obspars["Omega_field"] *= u.deg**2
    obspars["a_FG"] *= u.Mpc**-1

    global fiducialcosmoparams
    fiducialcosmoparams = cosmopars

    global fiducialastroparams
    fiducialastroparams = astropars

    # seperate the nuiscance-like cosmology parameters
    # add new ones here
    global nuiscance_like_params_names
    nuiscance_like_params_names = ["f_NL",
                                   "slope_ncdm",
                                   "k_cut_ncdm"]

    global fiducialnuiscancelikeparams
    fiducialnuiscancelikeparams = dict()
    for key in cosmopars:
        if key in nuiscance_like_params_names:
            fiducialnuiscancelikeparams[key]=cosmopars.pop(key)

    global fiducialfullcosmoparams
    fiducialfullcosmoparams = {**fiducialcosmoparams, **fiducialnuiscancelikeparams}

    global fiducialcosmo
    fiducialcosmo = cosmology.cosmo_functions(cosmopars=cosmopars,
                                              nuiscance_like=fiducialnuiscancelikeparams,
                                              input=input_type,
                                              )

    global fiducialastro
    fiducialastro = astro.astro_functions(cosmopars, astropars)

    global fiducialBAOparams
    fiducialBAOparams = BAOpars
