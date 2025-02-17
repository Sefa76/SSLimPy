# This class defines global variables that will not change through
# the computation of one fisher matrix/ chain/ etc
import os
import yaml
from astropy import units as u

def init(
    settings_dict=dict(),
    camb_yaml_file=None,
    class_yaml_file=None,
    obspars_dict=dict(),
    cosmopars=dict(),
    halopars=dict(),
    astropars=dict(),
    BAOpars=dict(),
):
    """This class is to handle the configuration for the computation as well as the fiducial parameters. It then gives access to all global variables

    Parameters
    ----------
    settings        : dict
                      A dictionary containing all additional neccessary options for the computation
    obspars_dict    : dict
                      A dictionary containing survey specifications
    camb_yaml_file  : String
                      A path containing the configuration of the EBS camb. Defaults to the default
    class_yaml_file : String
                      A path containing the configuration of the EBS class. Defaults to the default
    cosmopars       : dict, optional
                      A dictionary containing the fiducial cosmological parameters
    astropars       : dict, optional
                      A dictionary containing the fiducial astrophysical parameters
    BAOpars         : dict, optional
                      A dictionary containing the fiducial BAO toy-model parameters for BAO-only analysis
    """

    # Set global defaults for the settings dictionary
    global settings
    settings = settings_dict

    # Cosmology numerics
    settings.setdefault("nonlinearMatpow", True)
    settings.setdefault("share_delta_neff", True)
    settings.setdefault("LP_rescale_ini_As", 2.1e-9)
    settings.setdefault("LP_rescale_boost", 2)
    settings.setdefault("cosmo_model", "LCDM")
    settings.setdefault("code", "camb")
    settings.setdefault("h-units", False)
    settings.setdefault("do_pheno_ncdm", False)

    # Savgol numerics
    settings.setdefault("savgol_polyorder", 3)
    settings.setdefault("savgol_width", 1.358528901113328)
    settings.setdefault("savgol_internalsamples", 800)
    settings.setdefault("savgol_internalkmin", 0.001)

    # Pk numerics
    settings.setdefault("k_kind", "log")
    settings.setdefault("kmin", 1.0e-3 * u.Mpc**-1)
    settings.setdefault("kmax", 50 * u.Mpc**-1)
    settings.setdefault("nk", 200)
    settings.setdefault("zmin", 0)
    settings.setdefault("zmax", 5)
    settings.setdefault("nz", 32)
    settings.setdefault("nmu", 128)
    settings.setdefault("downsample_conv_q", 1)
    settings.setdefault("downsample_conv_muq", 8)
    settings.setdefault("nnodes_legendre", 9)

    # Pk specifications
    settings.setdefault("do_Jysr", False)
    settings.setdefault("fix_cosmo_nl_terms", True)

    # Pk contributions
    settings.setdefault("QNLpowerspectrum", True)
    settings.setdefault("TracerPowerSpectrum", "matter")
    settings.setdefault("do_RSD", True)
    settings.setdefault("nonlinearRSD", True)
    settings.setdefault("FoG_damp", "Lorentzian")
    settings.setdefault("halo_model_PS", False)
    settings.setdefault("Smooth_resolution", True)
    settings.setdefault("Smooth_window", False)

    # VID numerics
    settings.setdefault("Tmin_VID", 1e-2 * u.uK)
    settings.setdefault("Tmax_VID", 100 * u.uK)
    settings.setdefault("fT0_min", 1e-5 * u.uK**-1)
    settings.setdefault("fT0_max", 1e4 * u.uK**-1)
    settings.setdefault("fT_min", 1e-5 * u.uK**-1)
    settings.setdefault("fT_max", 1e5 * u.uK**-1)
    settings.setdefault("nfT0", 1000)
    settings.setdefault("sigma_PT_stable", 0.0 * u.uK)
    settings.setdefault("nT", int(2**18))
    settings.setdefault("smooth_VID", True)
    settings.setdefault("Nbin_hist", 100)
    settings.setdefault("linear_VID_bin", False)
    settings.setdefault("subtract_VID_mean", False)
    settings.setdefault("Lsmooth_tol", 7)
    settings.setdefault("T0_Nlogsigma", 4)
    settings.setdefault("n_leggauss_nodes_FT", "../nodes1e5.txt")
    settings.setdefault("n_leggauss_nodes_IFT", "../nodes1e4.txt")

    # Output settings
    settings.setdefault("verbosity", 1)
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
            file_content_camb = yaml.safe_load(
                open(
                    os.path.join(
                        file_location, "../../input/solver_files/camb_default.yaml"
                    )
                )
            )
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
            file_content_class = yaml.safe_load(
                open(
                    os.path.join(
                        file_location, "../../input/solver_files/class_default.yaml"
                    )
                )
            )
        boltzmann_classpars = file_content_class

    global fiducialcosmoparams
    fiducialcosmoparams = cosmopars

    # seperate the nuiscance-like cosmology parameters
    # add new ones here
    global nuiscance_like_params_names
    nuiscance_like_params_names = ["f_NL", "slope_ncdm", "k_cut_ncdm"]

    global fiducialnuiscancelikeparams
    fiducialnuiscancelikeparams = dict()
    for key in cosmopars:
        if key in nuiscance_like_params_names:
            fiducialnuiscancelikeparams[key] = cosmopars.pop(key)

    global fiducialfullcosmoparams
    fiducialfullcosmoparams = {**fiducialcosmoparams, **fiducialnuiscancelikeparams}

    global fiducialhaloparams
    fiducialhaloparams = halopars

    global fiducialspecparams
    fiducialspecparams = obspars_dict

    global fiducialastroparams
    fiducialastroparams = astropars

    initialize_fiducialcosmo()
    initialize_fiducialhalomodel()
    initialize_fiducialspecs()
    initialize_fiducialastro()

    global fiducialBAOparams
    fiducialBAOparams = BAOpars

def initialize_fiducialcosmo():
    from SSLimPy.cosmology import cosmology
    global fiducialcosmo

    fiducialcosmo = cosmology.CosmoFunctions(
        cosmopars=fiducialcosmoparams,
        nuiscance_like=fiducialnuiscancelikeparams,
        input_type=input_type,
    )

def initialize_fiducialhalomodel():
    from SSLimPy.cosmology import halo_model
    global fiducialhalomodel

    fiducialhalomodel = halo_model.HaloModel(
        cosmo= fiducialcosmo,
        halopars= fiducialhaloparams,
    )

def initialize_fiducialspecs():
    from SSLimPy.interface import survey_specs
    global fiducialspecs

    fiducialspecs = survey_specs.SurveySpecifications(
        obspars= fiducialspecparams,
        cosmo= fiducialcosmo,
    )

def initialize_fiducialastro():
    from SSLimPy.cosmology import astro
    global fiducialastro

    fiducialastro = astro.AstroFunctions(
        halomodel= fiducialhalomodel,
        survey_specs= fiducialspecs,
        astropars= fiducialastroparams
    )
