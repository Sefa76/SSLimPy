import sys
from warnings import warn
from copy import copy

from numpy import atleast_1d

sys.path.append("../")
import SSLimPy.interface.config as cfg
import SSLimPy.interface.updater as updater

class sslimpy:
    def __init__(
        self,
        settings_dict = dict(),
        camb_yaml_file = None,
        class_yaml_file = None,
        specifications = None,
        cosmopars = dict(),
        astropars = dict(),
        BAOpars = dict(),
        freepars = dict(),
    ):

        print("#--------------------------------------------------#")
        print("")
        print("  SSSSS   SSSSS  L       i            PPPP   y    y ")
        print(" S     S S     S L            m   m   P    P y   y  ")
        print(" S       S       L     iii   m m m m  P    P  y y   ")
        print("  SSSSS   SSSSS  L       i   m  m  m  PPPP     y    ")
        print("       S       S L       i   m     m  P        y    ")
        print(" S     S S     S L       i   m     m  P       y     ")
        print("  SSSSS   SSSSS  LLLLL iiiii m     m  P      y      ")
        print("")
        print("#--------------------------------------------------#")
        sys.stdout.flush()

        cfg.init(settings_dict = settings_dict,
        camb_yaml_file = camb_yaml_file,
        class_yaml_file = class_yaml_file,
        specifications = specifications,
        cosmopars = cosmopars,
        astropars = astropars,
        BAOpars= BAOpars,
        )

        self.settings = cfg.settings
        self.fiducialcosmo = cfg.fiducialcosmo
        self.fiducialcosmoparams = cfg.fiducialcosmoparams

        self.fiducialastro = cfg.fiducialastro
        self.fiducialastroparams = cfg.fiducialastroparams

        self.obspars = cfg.obspars
        self.vidpars = cfg.vidpars
        self.output = atleast_1d(cfg.settings["output"])

        ### save very fist cosmology ###
        self.curent_cosmo = copy(self.fiducialcosmo)
        self.curent_astro = copy(self.fiducialastro)

        ### TEXT VOMIT ###
        if cfg.settings["verbosity"]>1:
            self.recap_options()
        ##################

    def compute(self, cosmopars, astropars, BAOpars):
        outputdict = {obs: None for obs in self.output}

        for obs in self.output:
            if obs=="Power spectrum":
                outputdict[obs] = self._compute_ps(cosmopars, astropars, BAOpars)
            if obs=="Covaraiance":
                outputdict[obs] = self._compute_cov(cosmopars, astropars, BAOpars)
            else:
                warn("Output {} asked for not recognised \n Skiped".format(obs))
        
        print("Done!")
        return outputdict

    def _compute_ps(self, cosmopars, astropars, BAOpars):
        from SSLimPy.LIMsurvey.PowerSpectra import PowerSpectra
        cosmo = updater.update_cosmo(self.curent_cosmo,
                                     cosmopars
                                     )

        astro = updater.update_astro(self.curent_cosmo,
                                     cosmopars,
                                     self.curent_astro,
                                     astropars,
                                     updated_cosmo=cosmo)

        return PowerSpectra(cosmo,astro,BAOpars)


    def _compute_cov(self, cosmopars, astropars, BAOpars):
        from SSLimPy.LIMsurvey.Covariance import Covariance
        ps = self._compute_ps(cosmopars,astropars,BAOpars)
        return Covariance(ps)

    def recap_options(self):
        """This will print all the selected options into the standard output"""
        print("")
        print("----------RECAP OF SELECTED OPTIONS--------")
        print("")
        print("Settings:")
        for key in cfg.settings:
            print("   " + key + ": {}".format(cfg.settings[key]))
        print("")
        print("Observational Parameters:")
        for key in cfg.obspars:
            print("   " + key + ": {}".format(cfg.obspars[key]))
        print("VID Parameters:")
        for key in cfg.vidpars:
            print("   " + key + ": {}".format(cfg.vidpars[key]))

