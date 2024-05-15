from astropy import units as u
from astropy import constants as c

import numpy as np
from SSLimPy.interface import config as cfg

class PowerSpectra:
    def __init__(self, cosmology, astro):
        self.cosmology = cosmology
        self.astro = astro

        #################################
        # Properties of target redshift #
        #################################
        self.nu = self.astro.nu
        self.z = self.astro.z

        if cfg.settings["do_Jysr"]:
            x = c.c/(4.*np.pi*self.nu*self.H*(1.*u.sr))
            self.CLT = x.to(u.Jy*u.Mpc**3/(u.Lsun*u.sr))
        else:
            x = c.c**3*(1+self.z)**2/(8*np.pi*c.k_B*self.nu**3*self.H)
            self.CLT = x.to(u.uK*u.Mpc**3/u.Lsun)

        #########################################
        # Masses, luminosities, and wavenumbers #
        #########################################
        self.M = astro.M
        self.L = astro.L
        if "log" in cfg.settings["k_kind"]:
            self.k_edge = np.geomspace(cfg.settings["kmin"],cfg.settings["kmax"],cfg.settings["nk"])
        else:
            self.k_edge = np.linspace(cfg.settings["kmin"],cfg.settings["kmax"],cfg.settings["nk"])
        self.k = (self.k_edge[:-1]+self.k_edge[1:])/2.
        self.dk = np.diff(self.k)
        self.mu_edge = np.linspace(-1,1,cfg.settings["nmu"])
        self.mu = (self.mu_edge[:-1]+self.mu_edge[1:])/2.
        self.dmu = np.diff(self.mu)
        self.k_par = self.k[:,None] * self.mu[None,:]
        self.k_perp = self.k[:,None] * np.sqrt(1 - np.power(self.mu[None,:],2))


    def c_NFW(self):
        pass
