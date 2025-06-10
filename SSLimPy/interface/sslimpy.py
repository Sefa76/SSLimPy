import sys
from copy import copy
from numpy import atleast_1d
from SSLimPy.interface.config import Configuration
from SSLimPy.interface import updater

class sslimpy:
    def __init__(
        self,
        settings_dict = dict(),
        camb_yaml_file = None,
        class_yaml_file = None,
        obspars_dict = dict(),
        cosmopars = dict(),
        astropars = dict(),
        BAOpars = dict(),
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
            astropars = astropars,
            BAOpars= BAOpars,
        )

        self.settings = self.cfg.settings
        self.fiducialcosmo = self.cfg.fiducialcosmo
        self.fiducialcosmoparams = self.cfg.fiducialcosmoparams

        self.fiducialastro = self.cfg.fiducialastro
        self.fiducialastroparams = self.cfg.fiducialastroparams

        self.output = atleast_1d(self.cfg.settings["output"])

        ### save very fist cosmology ###
        self.curent_astro = self.fiducialastro

        ### TEXT VOMIT ###
        if self.cfg.settings["verbosity"]>1:
            self.recap_options()
        ##################

    def compute(self, cosmopars, halopars, obspars, astropars, BAOpars=dict(), pobs_settings=dict(), output=None):
        """Main interface to compute the different SSLimPy outputs

        Inputs the different SSLimPy output options.
        """

        if not output:
            output = self.output
        outputdict = {}

        if "Power spectrum" in output:
            self._compute_ps(cosmopars, halopars, astropars, obspars, BAOpars, pobs_settings, outputdict, self.cfg)

        if "Covariance" in output:
            self._compute_cov(cosmopars, halopars, astropars, obspars, BAOpars, pobs_settings, outputdict, self.cfg)
        return outputdict

    def _compute_ps(self, cosmopars, halopars, astropars, obspars, BAOpars, pobs_settings, outputdict, configuration):
        from SSLimPy.LIMsurvey.power_spectrum import PowerSpectra
        self.curent_astro = updater.update_astro(
            self.curent_astro, cosmopars, halopars, astropars, obspars, configuration
        )
        outputdict["Power spectrum"] = PowerSpectra(self.curent_astro, BAOpars, pobs_settings)

    def _compute_cov(self, cosmopars, halopars, astropars, obspars, BAOpars, pobs_settings, outputdict, configuration):
        from SSLimPy.LIMsurvey.covariance import Covariance
        if "Power spectrum" in outputdict:
            pass
        else:
            self._compute_ps(cosmopars, halopars, astropars, obspars, BAOpars, pobs_settings, outputdict, configuration)
        outputdict["Covariance"] = Covariance(outputdict["Power spectrum"])


    def recap_options(self):
        """This will print all the selected options into the standard output"""
        print("")
        print("----------RECAP OF SELECTED OPTIONS--------")
        print("")
        print("Settings:")
        for key in self.cfg.settings:
            print("   " + key + ": {}".format(self.cfg.settings[key]))

