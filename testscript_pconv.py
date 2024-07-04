import SSLimPy.interface.sslimpy as sslimpy
import SSLimPy.LIMsurvey.PowerSpectra as PowerSpectra
import SSLimPy.LIMsurvey.Covariance as Covariance
import astropy.units as u

import numpy as np
import matplotlib.pyplot as plt

cosmodict = {"h":0.67,"Omegam":0.32,"Omegab":0.05,"As":2.1e-9, "mnu":0.06}
settings = {"code":"class"}

Asslimpy = sslimpy.sslimpy(settings_dict=settings,
                           cosmopars=cosmodict)

cosmo = Asslimpy.fiducialcosmo
astro = Asslimpy.fiducialastro

Pobs = PowerSpectra.PowerSpectra(cosmo,astro)
cov = Covariance.Covariance(cosmo,Pobs)

Pconv = cov.convolved_Pk()
plt.loglog(Pobs.k, Pconv[:,::16,0])
plt.xlabel(r"k $[\mathrm{Mpc}^{-1}]$")
plt.ylabel(r"P $[\mu \mathrm{K}^2 \mathrm{Mpc}^{3}]$")
plt.savefig("Pcovariance.pdf")
