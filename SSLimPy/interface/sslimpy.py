import sys
from copy import copy
from numpy import atleast_1d
import SSLimPy.interface.config as cfg
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

        print("#---------------------------------------------------#")
        print("")
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

        cfg.init(settings_dict = settings_dict,
        camb_yaml_file = camb_yaml_file,
        class_yaml_file = class_yaml_file,
        obspars_dict = obspars_dict,
        cosmopars = cosmopars,
        astropars = astropars,
        BAOpars= BAOpars,
        )

        self.settings = cfg.settings
        self.fiducialcosmo = cfg.fiducialcosmo
        self.fiducialcosmoparams = cfg.fiducialcosmoparams

        self.fiducialastro = cfg.fiducialastro
        self.fiducialastroparams = cfg.fiducialastroparams

        self.output = atleast_1d(cfg.settings["output"])

        ### save very fist cosmology ###
        self.curent_astro = copy(self.fiducialastro)

        ### TEXT VOMIT ###
        if cfg.settings["verbosity"]>1:
            self.recap_options()
        ##################

    def compute(self, cosmopars, halopars, obspars, astropars, BAOpars=None, output=None):
        """Main interface to compute the different SSLimPy outputs

        Inputs the different SSLimPy output options.
        """

        if not output:
            output = self.output
        outputdict = {}

        if "Power spectrum" in output:
            self._compute_ps(cosmopars, halopars, astropars, obspars, BAOpars, outputdict)

        if "Covariance" in output:
            self._compute_cov(cosmopars, halopars, astropars, obspars, BAOpars, outputdict)
        return outputdict

    def _compute_ps(self, cosmopars, halopars, astropars, obspars, BAOpars, outputdict):
        from SSLimPy.SSLimPy.LIMsurvey.power_spectrum import PowerSpectra
        self.curent_astro = updater.update_astro(
            self.current_astro, cosmopars, halopars, astropars, obspars,
        )
        outputdict["Power spectrum"] = PowerSpectra(self.curent_astro, BAOpars)

    def _compute_cov(self, cosmopars, halopars, astropars, obspars, BAOpars, outputdict):
        from SSLimPy.SSLimPy.LIMsurvey.covariance import Covariance
        if "Power spectrum" in outputdict:
            pass
        else:
            self._compute_ps(cosmopars, halopars, astropars, obspars, BAOpars, outputdict)
        outputdict["Covariance"] = Covariance(outputdict["Power spectrum"])


    def recap_options(self):
        """This will print all the selected options into the standard output"""
        print("")
        print("----------RECAP OF SELECTED OPTIONS--------")
        print("")
        print("Settings:")
        for key in cfg.settings:
            print("   " + key + ": {}".format(cfg.settings[key]))

