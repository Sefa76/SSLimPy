import sys
from copy import copy
from numpy import atleast_1d
from SSLimPy.interface.config import Configuration
from SSLimPy.interface import updater

class SSLimPy:
    def __init__(
        self,
        settings_dict = dict(),
        camb_yaml_file = None,
        class_yaml_file = None,
        cosmopars = dict(),
        halopars = dict(),
        astropars = dict(),
        obspars_dict = dict(), 
    ):

        print("  █████   █████  █       █            █████   █    █ ")
        print(" █     █ █     █ █            █   █   █    █  █   █  ")
        print(" █       █       █     ███   █ █ █ █  █    █   █ █   ")
        print("  █████   █████  █       █   █  █  █  █████     █    ")
        print("       █       █ █       █   █     █  █        █     ")
        print(" █     █ █     █ █       █   █     █  █       █      ")
        print("  █████   █████  █████ █████ █     █  █      █       ")
        print("")
        print("#---------------------------------------------------#")
        sys.stdout.flush()

        self.cfg = Configuration(
            settings_dict = settings_dict,
            camb_yaml_file = camb_yaml_file,
            class_yaml_file = class_yaml_file,
            obspars_dict = obspars_dict,
            cosmopars = cosmopars,
            halopars = halopars,
            astropars = astropars,
        )

        self.settings = self.cfg.settings

        # Save fiducial parameters for better availability
        self.fiducialcosmoparams = self.cfg.fiducialcosmoparams
        self.fiducialhaloparams = self.cfg.fiducialhaloparams
        self.fiducialspecparams = self.cfg.fiducialspecparams
        self.fiducialastroparams = self.cfg.fiducialastroparams
        # The fiducial cosmology and survey_specs are saved in the Configurations object 

        # Save very first cosmology
        self.current_cosmology = copy(self.cfg.fiducialcosmo)
        self.current_halomodel = copy(self.cfg.fiducialhalomodel)
        self.current_survey_specs = copy(self.cfg.fiducialspecs)
        self.current_astro = copy(self.cfg.fiducialastro)
        # These will get updated as the code runs

        self.output = atleast_1d(self.cfg.settings["output"])

        ### TEXT VOMIT ###
        if self.cfg.settings["verbosity"]>1:
            self.recap_options()
        ##################

    def compute(self, cosmopars, halopars, astropars, obspars, BAOpars=dict(), pobs_settings=dict(), output=None):
        """Main interface to compute the different SSLimPy outputs

        Inputs the different SSLimPy output options.
        """

        if not output:
            output = self.output
        outputdict = {}

        if "Power spectrum" in output:
            self._compute_ps(cosmopars, halopars, astropars, obspars, self.cfg, BAOpars, pobs_settings, outputdict)

        if "Covariance" in output:
            self._compute_cov(cosmopars, halopars, astropars, obspars, self.cfg, BAOpars, pobs_settings, outputdict)
        return outputdict

    def _compute_ps(self, cosmopars, halopars, astropars, obspars, configuration, BAOpars, pobs_settings, outputdict):
        from SSLimPy.LIMsurvey.power_spectrum import PowerSpectra
        astro = updater.update_astro(
            self.current_astro, cosmopars, halopars, astropars, obspars, configuration
        )
        self._update_current(astro)
        outputdict["Power spectrum"] = PowerSpectra(self.current_astro, BAOpars, pobs_settings)

    def _compute_cov(self, cosmopars, halopars, astropars, obspars, configuration, BAOpars, pobs_settings, outputdict):
        from SSLimPy.LIMsurvey.covariance import Covariance
        if "Power spectrum" in outputdict:
            pass
        else:
            self._compute_ps(cosmopars, halopars, astropars, obspars, configuration, BAOpars, pobs_settings, outputdict)
        outputdict["Covariance"] = Covariance(outputdict["Power spectrum"])

    def _update_current(self, astro):
        self.current_astro = astro
        self.current_survey_specs = astro.survey_specs
        self.current_halomodel = astro.halomodel
        self.current_cosmology = astro.cosmology

    def recap_options(self):
        """This will print all the selected options into the standard output"""
        print("")
        print("----------RECAP OF SELECTED OPTIONS--------")
        print("")
        print("Settings:")
        for key in self.cfg.settings:
            print("   " + key + ": {}".format(self.cfg.settings[key]))

