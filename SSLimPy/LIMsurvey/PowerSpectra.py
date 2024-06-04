from astropy import units as u
from astropy import constants as c

from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.special import sici

import numpy as np
from SSLimPy.interface import config as cfg


class PowerSpectra:
    def __init__(self, cosmology, astro):
        self.cosmology = cosmology
        self.fiducialcosmo = cfg.fiducialcosmo
        self.astro = astro

        #################################
        # Properties of target redshift #
        #################################
        self.nu = self.astro.nu
        self.z = self.astro.z
        self.H = self.cosmology.Hubble(self.z, physical=True)

        if cfg.settings["do_Jysr"]:
            x = c.c / (4.0 * np.pi * self.nu * self.H * (1.0 * u.sr))
            self.CLT = x.to(u.Jy * u.Mpc**3 / (u.Lsun * u.sr))
        else:
            x = c.c**3 * (1 + self.z) ** 2 / (8 * np.pi * c.k_B * self.nu**3 * self.H)
            self.CLT = x.to(u.uK * u.Mpc**3 / u.Lsun)

        #########################################
        # Masses, luminosities, and wavenumbers #
        #########################################
        self.M = astro.M
        self.L = astro.L
        if "log" in cfg.settings["k_kind"]:
            self.k_edge = np.geomspace(
                cfg.settings["kmin"], cfg.settings["kmax"], cfg.settings["nk"]
            )
        else:
            self.k_edge = np.linspace(
                cfg.settings["kmin"], cfg.settings["kmax"], cfg.settings["nk"]
            )
        self.k = (self.k_edge[:-1] + self.k_edge[1:]) / 2.0
        self.dk = np.diff(self.k)
        self.mu_edge = np.linspace(-1, 1, cfg.settings["nmu"])
        self.mu = (self.mu_edge[:-1] + self.mu_edge[1:]) / 2.0
        self.dmu = np.diff(self.mu)
        self.k_par = self.k[:, None] * self.mu[None, :]
        self.k_perp = self.k[:, None] * np.sqrt(1 - np.power(self.mu[None, :], 2))

        self.c_NFW = self.prepare_c_NFW()

    def prepare_c_NFW(self):
        """
        concentration-mass relation for the NFW profile.
        Following Diemer & Joyce (2019)
        c = R_delta / r_s (the scale radius, not the sound horizon)
        """
        zvec = self.cosmology.results.zgrid
        Mvec = self.astro.M

        # fit parameters
        kappa = 0.42
        a0 = 2.37
        a1 = 1.74
        b0 = 3.39
        b1 = 1.82
        ca = 0.2

        # Compute the effective slope of the growth factor
        alpha_eff = self.cosmology.growth_rate(1e-4/u.Mpc,zvec)
        # Compute the effective slope to the power spectrum (as function of M)
        sigmaM = self.astro.sigmaM(Mvec, zvec)
        dsigmaM_dM = self.astro.dsigmaM_dM(Mvec, zvec)

        neff_at_R = -2.0 * 3.0 * Mvec[:, None] / sigmaM * dsigmaM_dM - 3.0

        # konvert to solar masses
        logMvec = np.log(Mvec.to(u.Msun).value)
        neff_inter = interp1d(logMvec, neff_at_R, fill_value="extrapolate", kind="linear", axis=0)
        neff = neff_inter(np.log(kappa**3 * Mvec.to(u.Msun).value))

        # Quantities for c
        A = a0 * (1.0 + a1 * (neff + 3))
        B = b0 * (1.0 + b1 * (neff + 3))
        C = 1.0 - ca * (1.0 - alpha_eff)
        nu = 1.686 / sigmaM
        arg = A / nu * (1.0 + nu**2 / B)

        # Compute G(x), with x = r/r_s, and evaluate c
        x = np.logspace(-3, 3, 256)
        g = np.log(1 + x) - x / (1.0 + x)
        G = x[None,None,:]/np.power(g[None,None,:],(5.0+neff[:,:,None])/6.0)

        c = np.zeros_like(arg)
        for iM, G_z_and_x in enumerate(G):
            for iz, G_x in enumerate(G_z_and_x):
                invG = interp1d(G_x, x, fill_value="extrapolate", kind="linear")
                c[iM,iz] = C[iz] * invG(arg[iM,iz])

        c_spline = RectBivariateSpline(logMvec,zvec,c)
        # restore units
        def cNFW_of_M_and_z(M,z):
            M = np.atleast_1d(M)
            z = np.atleast_1d(z)
            logM = np.log(M.to(u.Msun).value)
            return c_spline(logM[:,None],z[None,:])

        return cNFW_of_M_and_z

    def ft_NFW(self,k,M,z):
        '''
        Fourier transform of NFW profile, for computing one-halo term
        '''
        k = np.atleast_1d(k)
        M = np.atleast_1d(M)
        z = np.atleast_1d(z)

        #Radii of the SO collapsed (assuming 200*rho_crit)
        Delta = 200.
        rho_crit = self.astro.rho_crit
        R_NFW = (3.*M/(4.*np.pi*Delta*rho_crit))**(1./3.)

        #get characteristic radius
        c = self.c_NFW(M,z)[None,:,:]
        r_s = R_NFW[None,:,None]/c
        gc = np.log(1+c)-c/(1.+c)
        #argument: k*rs
        x = (k[:,None,None]*r_s).to(1).value

        si_x, ci_x = sici(x)
        si_cx, ci_cx = sici((1.+c)*x)
        u_km = (np.cos(x)*(ci_cx - ci_x) +
                  np.sin(x)*(si_cx - si_x) - np.sin(c*x)/((1.+c)*x))
        return np.squeeze(u_km/gc)
    
    ###############
    # De-Wiggling #
    ###############
    def sigmavNL(self, z, mu):
        z = np.atleast_1d(z)
        mu = np.atleast_1d(mu)
    
        if cfg.settings["fix_cosmo_nl_terms"]:
            cosmo = self.fiducialcosmo
        else:
            cosmo = self.cosmology

        if cfg.settings["nonlinearSwitch"]:
            f0 = np.atleast_1d(np.power(cosmo.P_ThetaTheta_Moments(z, 0), 2))[:,None]
            f1 = np.atleast_1d(np.power(cosmo.P_ThetaTheta_Moments(z, 1), 2))[:,None]
            f2 = np.atleast_1d(np.power(cosmo.P_ThetaTheta_Moments(z, 2), 2))[:,None]
            sv = np.sqrt(f0 + 2 * mu[None,:]**2 * f1 + mu[None,:]**2 * f2)
        else:
            sv = 0
        
        return np.squeeze(sv)

    def dewiggled_pdd(self, k, z, mu):
        """ "
        Calculates the normalized dewiggled powerspectrum

        Args:
        z : float
            The redshift value.
        k : float
            The wavenumber in Mpc^-1.
        mu : float
            The cosine of angle between the wavevector and the line-of-sight direction.

        Retruns:
            The dewiggled powerspectrum calculated with the Zeldovic approximation.
            If the config asks for only linear spectrum
        """
        k = np.atleast_1d(k)
        z = np.atleast_1d(z)
        mu = np.atleast_1d(mu)

        gmudamping = np.reshape(self.sigmavNL(z, mu) ** 2, (*z.shape, *mu.shape))

        P_dd = self.cosmology.matpow(k, z, tracer=cfg.settings["TracerPowerSpectrum"], nonlinear=False)
        P_dd_NW = self.cosmology.nonwiggle_pow(k, z, tracer=cfg.settings["TracerPowerSpectrum"], nonlinear=False)
        P_dd = np.reshape(P_dd,(*k.shape,*z.shape))
        P_dd_NW = np.reshape(P_dd_NW,(*k.shape,*z.shape))

        P_dd_DW = ( P_dd[:, :, None] * np.exp(-gmudamping[None,:,:] * k[:, None, None]**2) 
                  + P_dd_NW[:, :, None] * (1 - np.exp(-gmudamping[None,:,:] * k[:, None, None]**2)))
        return np.squeeze(P_dd_DW)
    
        
