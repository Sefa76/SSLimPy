from astropy import units as u
from astropy import constants as c

from time import time

import numpy as np
from SSLimPy.interface import config as cfg
from copy import copy

class Covariance:
    def __init__(self, cosmology, powerspectrum):
        self.cosmology = cosmology
        self.powerspectrum = powerspectrum       


    def L_side(self, zc, deltaz):
        zgrid = np.array([zc - 0.5 * deltaz, zc + 0.5 * deltaz])
        Lgrid = self.cosmology.comoving(zgrid)
        return Lgrid[1] - Lgrid[0]
