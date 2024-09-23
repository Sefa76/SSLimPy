from copy import copy

import astropy.units as u
import numpy as np

import SSLimPy.interface.config as cfg

"""
This class is for handeling all the cosmology-independent nuisance effects and corrections
"""


class nuiscance:
    def __init__(self, nuiscancepars, cosmology):
        self.nuscanceparams = copy(nuiscancepars)
        self.cosmology = cosmology
