from copy import copy
from time import time

import numpy as np
from astropy import units as u

from SSLimPy.cosmology.astro import AstroFunctions
from SSLimPy.utils.utils import *

class VoxelIntensity:

    def __init__(self, astro: AstroFunctions, vid_params:dict=dict()):
        self.cfg = astro.cfg

        # Load main cascade
        self.cosmology = astro.cosmology
        self.halomodel = astro.halomodel
        self.survey_specs = astro.survey_specs
        self.astro = astro

        self.tracer = self.halomodel.tracer

        self.vid_params = vid_params.copy()
        self._set_defaults_vid()

    def _set_defaults_vid(self):
        # Setting taken from lim
        # TODO: clean this up
        self.vid_params.setdefault("Tmin_VID", 1e-2 * u.uK)
        self.vid_params.setdefault("Tmax_VID", 100 * u.uK)
        self.vid_params.setdefault("fT0_min", 1e-5 * u.uK**-1)
        self.vid_params.setdefault("fT0_max", 1e4 * u.uK**-1)
        self.vid_params.setdefault("fT_min", 1e-5 * u.uK**-1)
        self.vid_params.setdefault("fT_max", 1e5 * u.uK**-1)
        self.vid_params.setdefault("nfT0", 1000)
        self.vid_params.setdefault("sigma_PT_stable", 0.0 * u.uK)
        self.vid_params.setdefault("nT", int(2**18))
        self.vid_params.setdefault("smooth_VID", True)
        self.vid_params.setdefault("Nbin_hist", 100)
        self.vid_params.setdefault("linear_VID_bin", False)
        self.vid_params.setdefault("subtract_VID_mean", False)
        self.vid_params.setdefault("Lsmooth_tol", 7)
        self.vid_params.setdefault("T0_Nlogsigma", 4)
        self.vid_params.setdefault("n_leggauss_nodes_FT", "../nodes1e5.txt")
        self.vid_params.setdefault("n_leggauss_nodes_IFT", "../nodes1e4.txt")