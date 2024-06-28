from astropy import units as u
from astropy import constants as c
from scipy.special import j1
from time import time

import numpy as np
from SSLimPy.interface import config as cfg
from copy import copy

class Covariance:
    def __init__(self, fiducialcosmology, powerspectrum):
        self.cosmology = fiducialcosmology
        self.powerspectrum = powerspectrum       

    def Lfield(self, z1, z2):
        zgrid = ([z1, z2])
        Lgrid = self.cosmology.comoving(zgrid)
        return Lgrid[1] - Lgrid[0]

    def Sfield(self, zc, Omegafield):
        r2 = np.power(self.cosmolgy.comoving(zc).to(u.Mpc), 2)
        sO = Omegafield.to(u.rad**2).value
        return r2*sO

    def convolved_Pk(self):
        """Convolves the Observed power spectrum with the survey volume
        
        This enters as the gaussian term in the final covariance
        """
        # Extract the pre-computed power spectrum
        k = self.powerspectrum.k
        mu = self.powerspectrum.mu
        z = self.powerspectrum.z
        Pk = self.powerspectrum.Pk_Obs

        # Get dummy wavevector q
        q = copy(k)
        muq = copy(mu)

        # The aztimuth angle  enters only like this
        # to the sum of k and q vetor
        deltaphi = np.linspace(-np.pi,np.pi,127)

        # related vectors
        # q muq deltaphi k mu z
        vq = q[:, None, None, None, None, None]
        vmuq = muq[None, :, None, None, None, None] 
        vdeltaphi = deltaphi[None, None, :, None, None, None]
        vk = k[None, None, None, :, None, None]
        vmu = mu[None, None, None, None, :, None]

        qparr = vq * vmuq
        qperp = vq * np.sqrt(1 - np.power(vmuq, 2))
        kparr = vk * vmu
        kperp = vk * np.sqrt(1 - np.power(vmu, 2))

        normkmq = np.sqrt(np.power(vk, 2)
                          + np.power(vq, 2)
                          - 2 * (kparr * qparr +
                                 kperp * qperp * np.cos(vdeltaphi)
                                 )
                          )
        mukmq = (kperp - qperp) / normkmq

        # Calculate dz from deltanu
        nu = self.powerspectrum.nu
        nuObs = self.powerspectrum.nuObs
        Delta_nu = cfg.obspars["Delta_nu"]
        z_min = (nu / (nuObs + Delta_nu / 2) - 1).to(1).value
        z_max = (nu / (nuObs - Delta_nu / 2) - 1).to(1).value

        # Construct W_survey (now just a cylinder)
        Lperp = np.sqrt(self.Sfield(z, cfg.obspars["Omega_field"])/np.pi)
        Wperp = 2 * np.pi * Lperp * j1((qperp * Lperp).to(1).value) / qperp
        Lparr = self.Lfield(z_min, z_max) 
        Wparr = 2 * np.sin((qparr * Lparr / 2).to(1).value) / qparr
        print(Wparr.shape)
        print(Wparr)


