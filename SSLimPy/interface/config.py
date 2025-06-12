# This class defines global variables that will not change through
# the computation of one fisher matrix/ chain/ etc
import os
import yaml
from astropy import units as u
from copy import copy

class Configuration:
    def __init__(
        self,
        settings_dict=dict(),
        camb_yaml_file=None,
        class_yaml_file=None,
        obspars_dict=dict(),
        cosmopars=dict(),
        halopars=dict(),
        astropars=dict(),
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
        """

        # Set global defaults for the settings dictionary
        self.settings = copy(settings_dict)

        # Cosmology numerics
        self.settings.setdefault("nonlinearMatpow", True)
        self.settings.setdefault("share_delta_neff", True)
        self.settings.setdefault("LP_rescale_ini_As", 2.1e-9)
        self.settings.setdefault("LP_rescale_boost", 2)
        self.settings.setdefault("cosmo_model", "LCDM")
        self.settings.setdefault("code", "camb")
        self.settings.setdefault("h-units", False)
        self.settings.setdefault("do_pheno_ncdm", False)

        # Savgol numerics
        self.settings.setdefault("savgol_polyorder", 3)
        self.settings.setdefault("savgol_width", 1.358528901113328)
        self.settings.setdefault("savgol_internalsamples", 800)
        self.settings.setdefault("savgol_internalkmin", 0.001)

        # Pk numerics
        self.settings.setdefault("k_kind", "log")
        self.settings.setdefault("kmin", 1.0e-3 * u.Mpc**-1)
        self.settings.setdefault("kmax", 50 * u.Mpc**-1)
        self.settings.setdefault("nk", 200)
        self.settings.setdefault("zmin", 0)
        self.settings.setdefault("zmax", 5)
        self.settings.setdefault("nz", 32)
        self.settings.setdefault("downsample_conv_q", 1)
        self.settings.setdefault("downsample_conv_muq", 8)
        self.settings.setdefault("nnodes_legendre", 9)

        # Pk specifications
        self.settings.setdefault("do_Jysr", False)
        self.settings.setdefault("fix_cosmo_nl_terms", True)

        # Pk contributions
        self.settings.setdefault("QNLpowerspectrum", True)
        self.settings.setdefault("TracerPowerSpectrum", "matter")
        self.settings.setdefault("do_RSD", True)
        self.settings.setdefault("nonlinearRSD", True)
        self.settings.setdefault("FoG_damp", "Lorentzian")
        self.settings.setdefault("halo_model_PS", False)
        self.settings.setdefault("Smooth_resolution", True)
        self.settings.setdefault("Smooth_window", False)

        # Pk FFTlog approximation
        self.settings.setdefault("Log-extrap", 10)
        self.settings.setdefault("LogN_modes", 10)


        # VID numerics
        self.settings.setdefault("Tmin_VID", 1e-2 * u.uK)
        self.settings.setdefault("Tmax_VID", 100 * u.uK)
        self.settings.setdefault("fT0_min", 1e-5 * u.uK**-1)
        self.settings.setdefault("fT0_max", 1e4 * u.uK**-1)
        self.settings.setdefault("fT_min", 1e-5 * u.uK**-1)
        self.settings.setdefault("fT_max", 1e5 * u.uK**-1)
        self.settings.setdefault("nfT0", 1000)
        self.settings.setdefault("sigma_PT_stable", 0.0 * u.uK)
        self.settings.setdefault("nT", int(2**18))
        self.settings.setdefault("smooth_VID", True)
        self.settings.setdefault("Nbin_hist", 100)
        self.settings.setdefault("linear_VID_bin", False)
        self.settings.setdefault("subtract_VID_mean", False)
        self.settings.setdefault("Lsmooth_tol", 7)
        self.settings.setdefault("T0_Nlogsigma", 4)
        self.settings.setdefault("n_leggauss_nodes_FT", "../nodes1e5.txt")
        self.settings.setdefault("n_leggauss_nodes_IFT", "../nodes1e4.txt")

        # Output settings
        self.settings.setdefault("verbosity", 1)
        self.settings.setdefault("output", [])

        # Load Boltzmann solver files
        self.input_type = self.settings["code"]

        file_location = os.path.dirname(__file__)
        if self.input_type == "camb":
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
            self.boltzmann_cambpars = file_content_camb

        if self.input_type == "class":
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
            self.boltzmann_classpars = file_content_class

        self.fiducialcosmoparams = copy(cosmopars)

        # seperate the nuiscance-like cosmology parameters
        # add new ones here
        self.nuiscance_like_params_names = ["f_NL", "slope_ncdm", "k_cut_ncdm"]

        self.fiducialnuiscancelikeparams = dict()
        for key in cosmopars:
            if key in self.nuiscance_like_params_names:
                self.fiducialnuiscancelikeparams[key] = self.fiducialcosmoparams.pop(key)

        self.fiducialfullcosmoparams = {
            **self.fiducialcosmoparams,
            **self.fiducialnuiscancelikeparams,
        }

        self.fiducialhaloparams = copy(halopars)

        self.fiducialspecparams = copy(obspars_dict)

        self.fiducialastroparams = copy(astropars)

        self.initialize_fiducialcosmo()
        self.initialize_fiducialhalomodel()
        self.initialize_fiducialspecs()
        self.initialize_fiducialastro()

    def initialize_fiducialcosmo(self):
        from SSLimPy.cosmology import cosmology

        self.fiducialcosmo = cosmology.CosmoFunctions(
            cfg = self,
            cosmopars=self.fiducialcosmoparams,
            nuiscance_like=self.fiducialnuiscancelikeparams,
            input_type=self.input_type,
        )

    def initialize_fiducialhalomodel(self):
        from SSLimPy.cosmology import halo_model

        self.fiducialhalomodel = halo_model.HaloModel(
            cosmo= self.fiducialcosmo,
            halopars= self.fiducialhaloparams,
        )

    def initialize_fiducialspecs(self):
        from SSLimPy.interface import survey_specs

        self.fiducialspecs = survey_specs.SurveySpecifications(
            obspars= self.fiducialspecparams,
            cosmo= self.fiducialcosmo,
        )

    def initialize_fiducialastro(self):
        from SSLimPy.cosmology import astro

        self.fiducialastro = astro.AstroFunctions(
            halomodel= self.fiducialhalomodel,
            survey_specs= self.fiducialspecs,
            astropars= self.fiducialastroparams
        )