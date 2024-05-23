from astropy import units as u
from astropy import constants as c

from scipy.interpolate import RectBivariateSpline, interp1d

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
        self.H = self.cosmology.Hubble(self.z,physical=True)

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


    def prepare_c_NFW(self):
        '''
        concentration-mass relation for the NFW profile.
        Following Diemer & Joyce (2019)
        c = R_delta / r_s (the scale radius, not the sound horizon)
        '''
        zvec = self.cosmology.results.zgrid
        Mvec = self.astro.M

        #Compute the effective slope of the growth factor
        alpha_eff = self.cosmology.f_growthrate(zvec)[None,:]
        #Compute the effective slope to the power spectrum (as function of M)
        sigmaM, dsigmaM_dM = self.astro.compute_sigmaM_funcs(Mvec,zvec)
        neff_at_R = -2.*3.*Mvec/sigmaM*dsigmaM_dM-3.
        neff = interp1d(np.log10(self.M.value),neff_at_R,fill_value='extrapolate',kind='linear',axis=0)(np.log10(kappa**3 * Mvec))

        #fit parameters
        kappa = 0.42
        a0 = 2.37
        a1 = 1.74
        b0 = 3.39
        b1 = 1.82
        ca = 0.2
        #Quantities for c   
        A = a0*(1.+a1*(neff+3))
        B = b0*(1.+b1*(neff+3))
        C = 1.-ca*(1.-alpha_eff)
        nu = 1.686/sigmaM
        arg = A/nu*(1.+nu**2/B)

        #Compute G(x), with x = r/r_s, and evaluate c
        x = np.logspace(-3,3,256)
        g = np.log(1+x)-x/(1.+x)

        c = np.zeros(len(Mvec))
        for iM in range(len(Mvec)):
            G = x/g**((5.+neff[iM])/6.)
            invG = interp1d(G,x,fill_value='extrapolate',kind='linear')
            c[iM] = C*invG(arg[iM])
            
        return interp1d(Mvec,c,fill_value='extrapolate',kind='cubic')(self.M.value)
