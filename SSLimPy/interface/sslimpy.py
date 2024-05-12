import sys
from warnings import warn

sys.path.append("../")
import SSLimPy.interface.config as cfg
class sslimpy:
    def __init__(
        self,
        settings_dict = dict(),
        camb_yaml_file = None,
        class_yaml_file = None,
        specifications = None,
        cosmopars = dict(),
        astropars = dict(),
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
        )

        self.settings = cfg.settings
        self.fiducialcosmo = cfg.fiducialcosmo
        self.fiducialcosmoparams = cfg.fiducialcosmoparams

        self.fiducialastro = cfg.fiducialastro
        self.fiducialastroparams = cfg.fiducialastroparams

        self.obspars = cfg.obspars
        self.vidpars = cfg.vidpars
        self.output = cfg.settings["output"]

        ### TEXT VOMIT ###
        if cfg.settings["verbosity"]>1:
            self.recap_options()
        ##################

    def compute(self, cosmology):
        if self.output is not None:
            for ops in self.output:
                if ops=="power spectrum":
                    self._compute_ps(cosmology)
                if ops=="Covaraiance":
                    self._compute_cov(cosmology)
                else:
                    warn("Output {} asked for not recognised \n Skiped".format(ops))
        print("Done!")

    def _compute_ps(self, cosmology):
        from SSLimPy.LIMsurvey.PowerSpectra import PowerSpectra
        return PowerSpectra()


    def _compute_cov(self, cosmology):
        from SSLimPy.LIMsurvey.Covariance import Covariance
        ps = self._compute_ps(cosmology)
        return Covariance(cosmology, ps)

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

