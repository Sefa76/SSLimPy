from copy import copy
from time import time
from typing import Tuple, Union

import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy.units import Quantity
from numpy.typing import DTypeLike, NDArray
from scipy.integrate import simpson
from scipy.interpolate import RectBivariateSpline, UnivariateSpline, interp1d
from scipy.special import j1, legendre, sici, spherical_jn

from SSLimPy.interface import config as cfg
from SSLimPy.utils.utils import convolve

k_types = Union[
    Quantity,
    NDArray,
    Tuple[Quantity, ...],
    Tuple[NDArray, ...]
]
mu_types = Union[
    DTypeLike,
    NDArray,
    Tuple[DTypeLike, ...],
    Tuple[NDArray, ...]
]

class PowerSpectra:
    def __init__(self, cosmology, astro, BAOpars=dict()):
        self.cosmology = cosmology
        self.fiducialcosmo = cfg.fiducialcosmo
        self.astro = astro
        self.BAOpars = copy(BAOpars)

        self.tracer = cfg.settings["TracerPowerSpectrum"]

        #################################
        # Properties of target redshift #
        #################################
        self.nu = cfg.obspars["nu"]
        self.nuObs = cfg.obspars["nuObs"]
        self.Delta_nu = cfg.obspars["Delta_nu"]
        self.z = cfg.z
        self.H = self.cosmology.Hubble(self.z, physical=True)

        #########################################
        # Masses, luminosities, and wavenumbers #
        #########################################
        self.M = astro.M
        self.L = astro.L
        if "log" in cfg.settings["k_kind"]:
            self.k_edge = np.geomspace(
                cfg.settings["kmin"], cfg.settings["kmax"], cfg.settings["nk"]+1
            )
        else:
            self.k_edge = np.linspace(
                cfg.settings["kmin"], cfg.settings["kmax"], cfg.settings["nk"]+1
            )
        self.k = (self.k_edge[:-1] + self.k_edge[1:]) / 2.0
        self.dk = np.diff(self.k)
        self.mu_edge = np.linspace(-1, 1, cfg.settings["nmu"]+1)
        self.mu = (self.mu_edge[:-1] + self.mu_edge[1:]) / 2.0
        self.dmu = np.diff(self.mu)
        self.k_par = self.k[:, None] * self.mu[None, :]
        self.k_perp = self.k[:, None] * np.sqrt(1 - np.power(self.mu[None, :], 2))

        ##########################
        # The LIM power spectrum #
        ##########################

        # Compute the power spectrum on the internal grid
        self.c_NFW = self.prepare_c_NFW()

        # self.compute_power_spectra()
        # self.compute_power_spectra_moments()

        # Create interpolationg fucuntion
        # self.Pk_obs_func = self.create_pk_interp()

    ###################
    # Halo Properties #
    ###################

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

    def halomoments(self,k: k_types ,z, mu: mu_types=None, moment: int = 1, bias_order:int = 0):
        """
        Computes the Luminosity weight halo profile
        In ML models this is equivalent to the n-halo-self-correlation terms
        In a LF model this will be off by a consant and should not be taken into an accurate physical model
        """
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        M = self.M
        z = np.atleast_1d(z)

        dn_dM = np.reshape(self.astro.halomassfunction(M,z),(*M.shape,*z.shape))
        L_of_M = np.reshape(self.astro.massluminosityfunction(M,z),(*M.shape,*z.shape))

        if bias_order == 0:
            bias = np.ones((*k.shape, *M.shape, *z.shape))
        elif bias_order == 1:
            bias = self.astro.restore_shape(self.astro.halobias(M,z,k=k),k,M,z)
        else:
            raise ValueError("Order of bias asked for does is not implemented: {}".format(bias_order))

        Fv = np.ones((*k.shape, *mu.shape, *M.shape, *z.shape))
        if self.astro.astroparams["v_of_M"]:
            Fv = np.reshape(self.astro.broadening_FT(k, mu, M, z),
                            (*k.shape, *mu.shape, *M.shape, *z.shape))

        normhaloprofile = np.reshape(self.ft_NFW(k, M, z),(*k.shape, *M.shape, *z.shape))

        integrnd = (
            dn_dM[None, None,:,:]
            * np.power(
                L_of_M[None, None,:,:]
                * normhaloprofile[:, None,:,:]
                * Fv,
                moment,
            )
            * bias[:,None,:,:]
        )
        hm_corr = np.trapz(integrnd,M,axis=2)
        return np.squeeze(hm_corr)

    ###############
    # De-Wiggling #
    ###############
    def sigmavNL(self, mu, z, BAOpars=dict()):
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)

        if cfg.settings["fix_cosmo_nl_terms"]:
            cosmo = self.fiducialcosmo
        else:
            cosmo = self.cosmology

        if "sigmav" in BAOpars:
            sigmav = np.atleast_1d(BAOpars["sigmav"])
            if len(sigmav) != len(z):
                raise ValueError("did not pass velocity dispertion for every z asked for")
            # scale independent f
            f_scaleindependent = np.atleast_1d(cosmo.growth_rate(1e-4/u.Mpc,z))
            sv2 = np.power(sigmav[None,:],2) * (
                1 
                - np.power(mu,2)[:,None]
                + np.power(mu,2)[:,None]
                * (1 + f_scaleindependent[None,:])**2
                )
        else:
            f0 = np.atleast_1d(np.power(cosmo.P_ThetaTheta_Moments(z, 0), 2))[:,None]
            f1 = np.atleast_1d(np.power(cosmo.P_ThetaTheta_Moments(z, 1), 2))[:,None]
            f2 = np.atleast_1d(np.power(cosmo.P_ThetaTheta_Moments(z, 2), 2))[:,None]
            sv2 = f0 + 2 * mu[None,:]**2 * f1 + mu[None,:]**2 * f2
        return np.squeeze(np.sqrt(sv2))

    def dewiggled_pdd(self, k, mu, z, BAOpars = dict()):
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
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)

        gmudamping = np.reshape(self.sigmavNL(mu, z, BAOpars=BAOpars) ** 2, (*mu.shape, *z.shape))

        P_dd = self.cosmology.matpow(k, z, tracer=self.tracer, nonlinear=False)
        P_dd_NW = self.cosmology.nonwiggle_pow(k, z, tracer=self.tracer, nonlinear=False)
        P_dd = np.reshape(P_dd,(*k.shape,*z.shape))
        P_dd_NW = np.reshape(P_dd_NW,(*k.shape,*z.shape))

        P_dd_DW = ( P_dd[:, None, :] * np.exp(-gmudamping[None,:,:] * k[:, None, None]**2)
                  + P_dd_NW[:, None, :] * (1 - np.exp(-gmudamping[None,:,:] * k[:, None, None]**2)))
        return np.squeeze(P_dd_DW)

    ################
    # BAO Features #
    ################
    def qparallel(self, z, BAOpars = dict()):
        """
        Function implementing q parallel of the Alcock-Paczynski effect
        If BAOpars is passed checks for alpha_par
        """
        z = np.atleast_1d(z)
        if "alpha_par" in BAOpars:
            q_par = np.atleast_1d(BAOpars["alpha_par"])
            if len(z) != len(q_par):
                raise ValueError("did not pass alpha_par parameters for every z asked for")
        else:
            fidTerm = self.fiducialcosmo.Hubble(z)
            cosmoTerm = self.cosmology.Hubble(z)
            q_par = fidTerm/cosmoTerm
        return q_par

    def qperpendicular(self, z, BAOpars = dict()):
        """
        Function implementing q perpendicular of the Alcock-Paczynski effect
        If BAOpars is passed checks for alpha_perp
        """
        z = np.atleast_1d(z)
        if "alpha_perp" in BAOpars:
            q_perp = np.atleast_1d(BAOpars["alpha_perp"])
            if len(z) != len(q_perp):
                raise ValueError("did not pass alpha_perp parameters for every z asked for")
        else:
            fidTerm = self.fiducialcosmo.angdist(z)
            cosmoTerm = self.cosmology.angdist(z)
            q_perp = cosmoTerm / fidTerm
        return q_perp

    def dragscale(self):
        """
        Function to fix the dragscale
        """
        fidTerm = self.fiducialcosmo.rs_drag()
        cosmoTerm = self.cosmology.rs_drag()
        return fidTerm/cosmoTerm

    def bias_term(self, z, k=None, mu=None, BAOpars=dict()):
        """
        Function to compute the bias term that enters the linear Kaiser formula
        If BAOpars is passed checks for bmean
        """
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)
        if "bmean" in BAOpars:
            bmean = np.atleast_1d(BAOpars["bmean"])
            if len(bmean) != len(z):
                raise ValueError("did not pass mean bias for every z asked for")

            # Does not contain correction for f_nl yet
            bmean = bmean[None,:]
            if "f_NL" in self.cosmology.fullcosmoparams:
                M = self.M
                fac = ((bmean-1)
                       / np.reshape(self.astro._b_of_M(M,z)-1,(*M.shape, *z.shape)))
                Delta_b = (self.astro.Delta_b(k, M, z) * fac[None,:,:])
                Delta_b = Delta_b[:,0,:] # Actually this is now M-independent

                bmean = bmean + Delta_b

            if "Tmean" in BAOpars:
                Tmean = np.atleast_1d(BAOpars["Tmean"])
                if len(Tmean) != len(z):
                    raise ValueError("did not pass mean Temperature for every z asked for")
                Tmean = Tmean[None, None, :]
            else:
                Tmean = self.astro.restore_shape(self.astro.Tmoments(z, k=k, mu=mu, moment=1), k, mu, z)

            Biasterm = (
                bmean[:, None, :]
                * Tmean
                * np.atleast_1d(self.cosmology.sigma8_of_z(z,tracer=self.tracer))[None,:]
            )
        else:
            Biasterm = (self.astro.restore_shape(self.astro.bavg(z, k=k, mu=mu), k, mu, z)
                        * self.astro.restore_shape(self.astro.Tmoments(z, k=k, mu=mu, moment=1), k, mu, z)
                        * np.atleast_1d(self.cosmology.sigma8_of_z(z,tracer=self.tracer))[None, None,:])
        return np.squeeze(Biasterm)

    def f_term(self,k,mu,z, BAOpars=dict()):
        """
        Function to compute the linear redshift space distortion that enters the linear Kaiser formula
        If BAOpars is passed checks for Tmean
        """
        k = np.atleast_1d(k)
        mu= np.atleast_1d(mu)
        z = np.atleast_1d(z)
        if "Tmean" in BAOpars:
            Tmean = np.atleast_1d(BAOpars["Tmean"])
            if len(Tmean) != len(z):
                raise ValueError("did not pass mean Temperature for every z asked for")
            Tmean = Tmean[None, None, :]
        else:
            Tmean = self.astro.restore_shape(self.astro.Tmoments(z, k=k, mu=mu,moment=1), k, mu, z)

        Lfs8 =  Tmean* np.reshape(self.cosmology.fsigma8_of_z(k,z,tracer=self.tracer), (*k.shape, 1, *z.shape))
        Kaiser_RSD = Lfs8*np.power(mu,2)[None,:,None]
        return np.squeeze(Kaiser_RSD)

    def Kaiser_Term(self,k,mu,z, BAOpars =dict()):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)
        bterm = self.astro.restore_shape(self.bias_term(z, k=k, mu=mu, BAOpars=BAOpars),k, mu, z)
        if "beta" in BAOpars:
            beta = np.atleast_1d(BAOpars["beta"])
            if len(beta) != len(z):
                raise ValueError("did not pass RSD amplitude for every z asked for")
            fterm = beta * np.power(mu[None,:,None], 2)
            linear_Kaiser = np.power(bterm * (1 + fterm), 2)
        else:
            fterm = np.reshape(self.f_term(k,mu,z,BAOpars=BAOpars),(*k.shape,*mu.shape,*z.shape))
            linear_Kaiser = np.power(bterm + fterm ,2)
        return np.squeeze(linear_Kaiser)

    def fingers_of_god(self, k, mu, z, BAOpars = dict()):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)

        if cfg.settings["fix_cosmo_nl_terms"]:
            cosmo = self.fiducialcosmo
        else:
            cosmo = self.cosmology

        if "sigmap" in BAOpars:
            # This quantitiy is different from the old lim one by a factor of f^2
            sigmap = np.atleast_1d(BAOpars["sigmap"])
            if len(sigmap) != len(z):
                raise ValueError("did not pass velocity dispertion for every z asked for")
            # scale independent f
            f_scaleindependent = cosmo.growth_rate(1e-4/u.Mpc,z)
            sp = sigmap*f_scaleindependent
        else:
            sp = np.atleast_1d(cosmo.P_ThetaTheta_Moments(z,moment=2))
        FoG_damp = cfg.settings["FoG_damp"]
        if FoG_damp == 'Lorentzian':
            FoG = np.power(1.+0.5*np.power(k[:,None,None]*mu[None,:,None]* sp[None,None,:],2),-2)
        elif FoG_damp == 'Gaussian':
            FoG = np.exp(-((k[:,None,None]*mu[None,:,None]*sp[None,None,:])**2.))
        elif FoG_damp == "ISTF_like":
            FoG = np.power(1+np.power(k[:,None,None]*mu[None,:,None]*sp[None,None,:],2),-1)

        return np.squeeze(FoG)

    ##################################
    # Line Broadening and Resolution #
    ##################################

    # Call these functions before applying the AP transformations, scale fixes, ect
    def sigma_parr(self, z, nu_obs):
        x = (cfg.obspars["dnu"] / nu_obs).to(1).value
        y = (1+z) / self.fiducialcosmo.Hubble(z)
        return x * y

    def sigma_perp(self, z):
        x = cfg.obspars["beam_FWHM"].to(u.rad).value / np.sqrt(8 * np.log(2))
        y = self.fiducialcosmo.angdist(z)*(1+z)
        return x * y

    def F_parr(self, k, mu, z, nu_obs):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)

        logF = -0.5 * np.power(k[:, None, None]
                               * mu[None, :, None]
                               * self.sigma_parr(z, nu_obs)[None, None, :],
                               2)

        return np.squeeze(np.exp(logF))

    def F_perp(self, k, mu, z):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)

        logF = -0.5 * np.power(k[:, None, None]
                               * np.sqrt(1 - mu[None, :, None]**2)
                               * self.sigma_perp(z)[None, None, :],
                               2)

        return np.squeeze(np.exp(logF))

    ##################################
    # Convolution and Survey Windows #
    ##################################

    def Lfield(self, z1, z2):
        zgrid = [z1, z2]
        Lgrid = self.fiducialcosmo.comoving(zgrid)
        return Lgrid[1] - Lgrid[0]

    def Sfield(self, zc, Omegafield):
        r2 = np.power(self.fiducialcosmo.comoving(zc).to(u.Mpc), 2)
        sO = Omegafield.to(u.rad**2).value
        return r2 * sO

    def Wsurvey(self, q, muq):
        """Compute the Fourier-transformed sky selection window function"""
        q = np.atleast_1d(q)
        muq = np.atleast_1d(muq)

        qparr = q[:, None, None] * muq[None, :, None]
        qperp = q[:, None, None] * np.sqrt(1 - np.power(muq[None, :, None], 2))

        # Calculate dz from deltanu
        nu = self.nu
        nuObs = self.nuObs
        Delta_nu = self.Delta_nu
        z = (nu / nuObs - 1).to(1).value
        z_min = (nu / (nuObs + Delta_nu / 2) - 1).to(1).value
        z_max = (nu / (nuObs - Delta_nu / 2) - 1).to(1).value

        # Construct W_survey (now just a cylinder)
        Sfield = self.Sfield(z, cfg.obspars["Omega_field"])
        Lperp = np.sqrt(Sfield / np.pi)
        Lparr = self.Lfield(z_min, z_max)
        Wperp = 2 * np.pi * Lperp * j1((qperp * Lperp).to(1).value) / qperp
        Wparr = Lparr * spherical_jn(0, (qparr * Lparr / 2).to(1).value)

        Wsurvey = Wperp * Wparr
        Vsurvey = simpson(
            q[:,None]**3
            * simpson(
                np.abs(Wsurvey)**2 / (2 * np.pi)**2, 
                muq,
                axis=1
            ),
            np.log(q.value),
            axis=0,
        ) * (Sfield * Lparr).unit
        return Wsurvey, Vsurvey

    def convolved_Pk(self):
        """Convolves the Observed power spectrum with the survey volume

        This enters as the gaussian term in the final covariance
        """
        # Extract the pre-computed power spectrum
        k = self.k
        mu = self.mu
        Pobs = self.Pk_Obs
        nz = Pobs.shape[-1]

        # Downsample q, muq and deltaphi
        nq = np.uint8(len(k) / cfg.settings["downsample_conv_q"])
        if "log" in cfg.settings["k_kind"]:
            q = np.geomspace(k[0], k[-1], nq)
        else:
            q = np.linspace(k[0], k[-1], nq)

        nmuq = np.uint8((len(mu)) / cfg.settings["downsample_conv_muq"])
        nmuq = nmuq + 1 - nmuq % 2
        muq = np.linspace(-1, 1, nmuq)
        muq = (muq[1:] + muq[:-1]) / 2.0
        deltaphi = np.linspace(-np.pi, np.pi, 2 * len(muq))

        # Obtain survey Window
        Wsurvey, Vsurvey = self.Wsurvey(q, muq)

        Pconv = np.empty(Pobs.shape)
        # Do the convolution for each redshift bin
        for iz in range(nz):
            Pconv[..., iz] = convolve(
                k.value,
                mu,
                q.value,
                muq,
                deltaphi,
                Pobs[:, :, iz].value,
                Wsurvey[:, :, iz].value,
            )
        
        self.Pk_true = copy(Pobs)
        self.Pk_Obs = Pconv *Pobs.unit / Vsurvey.value
        return self.Pk_Obs

    #######################
    # Power Spectra Terms #
    #######################

    def shotnoise(self, z, k=None, mu=None,BAOpars = dict()):
        """
        This function returns additional contributions to the Auto-power spectrum depending on what is asked for:
            - The Posoinian Shot noise
            - The One-Halo Term as scale dependent shot noise
            - Additional Shot noise from the BAOpars
            - Combinations of the fromer
        """
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)

        Ps = 0
        if "Pshot" in BAOpars:
            Pshot = np.atleast_1d(BAOpars["Pshot"])
            if len(Pshot) != len(z):
                raise ValueError("did not pass the shotnoise for every z asked for")
            Ps = Pshot[None, None,:]
        else:
            if cfg.settings["halo_model_PS"]:
                Ps = self.astro.restore_shape(self.halomoments(k, z, mu=mu, moment=2), k, mu, z)
                Ps *= self.astro.CLT(z)[None,None,:]**2
            else:
                Ps = self.astro.restore_shape(self.astro.Tmoments(z, k=k, mu=mu, moment=2), k, mu, z)
        return np.squeeze(Ps)

    def compute_power_spectra(self):
        """This function computes the full shape observed power spectrum
        with (nearly) fully vectorized function
        """
        #grid quantities
        k = copy(self.k)
        mu = copy(self.mu)
        z = copy(self.z)
        outputshape = (*k.shape,*mu.shape,*z.shape)
        if cfg.settings["verbosity"] >1:
            print("requested Pk shape:",outputshape)
            tstart = time()

        Fnu = np.ones(outputshape)
        if cfg.settings["Smooth_resolution"]:
            # The dampning from resolution is to be computed without any cosmolgy dependance
            F_parr = np.reshape(self.F_parr(k, mu, z, self.nuObs), outputshape)
            F_perp = np.reshape(self.F_perp(k, mu, z), outputshape)
            Fnu = F_parr*F_perp

        #fix units
        k *= self.dragscale()
        logkMpc = np.log(k.to(u.Mpc**-1).value)

        # Apply AP effect
        qparr = np.atleast_1d(self.qparallel(z, self.BAOpars))
        kparr_ap = k[:,None,None] * mu[None,:,None] * qparr[None,None,:]
        qperp = np.atleast_1d(self.qperpendicular(z, self.BAOpars))
        kperp_ap = k[:,None,None] *np.sqrt(1- np.power(mu[None,:,None],2)) * qperp[None,None,:]

        # Compute related quantities
        k_ap = np.sqrt(np.power(kparr_ap,2)+np.power(kperp_ap,2))
        mu_ap = kparr_ap/k_ap

        Pk_ap = np.zeros(outputshape)
        if cfg.settings["QNLpowerspectrum"]:
            # Obtain the normalized dewiggled power spectrum
            Pk_dw_grid = np.reshape(self.dewiggled_pdd(k,mu,z,BAOpars=self.BAOpars),outputshape)
            uI = Pk_dw_grid.unit
            logPk_dw_grid = np.log(Pk_dw_grid.value)
            for iz,zi in enumerate(z):
                interp_per_z = RectBivariateSpline(logkMpc,mu,logPk_dw_grid[:,:,iz])
                Pk_ap[:,:,iz] = np.exp(interp_per_z(np.log(k_ap[:,:,iz].to(u.Mpc**-1).value), mu_ap[:,:,iz], grid=False)) \
                                / np.atleast_1d(np.power(self.cosmology.sigma8_of_z(zi,tracer=self.tracer),2))[None,None,:]
        else:
            # Use linear power spectrum
            Pk_grid = np.reshape(self.cosmology.matpow(k,z),(*k.shape,*z.shape))
            uI = Pk_grid.unit
            logPk_grid = np.log(Pk_grid.value)
            for iz,zi in enumerate(z):
                interp_per_z = UnivariateSpline(logkMpc,logPk_grid[:,iz])
                Pk_ap[:,:,iz] = np.exp(interp_per_z(np.log(k_ap[:,:,iz].to(u.Mpc**-1).value))) \
                                / np.atleast_1d(np.power(self.cosmology.sigma8_of_z(zi,tracer=self.tracer),2))[None,None,:]
        Pk_ap *= uI

        if cfg.settings["verbosity"] >1:
            tPk = time()
            print("Power spectrum obtained in {} seconds".format(tPk-tstart))

        rsd_ap  = np.ones(outputshape)
        if cfg.settings["do_RSD"]:
            # Obtain redshiftspace distortions
            rsd_grid = self.Kaiser_Term(k,mu,z,BAOpars=self.BAOpars)
            if cfg.settings["nonlinearRSD"]:
                rsd_grid *= self.fingers_of_god(k,mu,z,BAOpars=self.BAOpars)
            rsd_grid = np.reshape(rsd_grid,outputshape)
            uI = rsd_grid.unit
            for iz, zi in enumerate(z):
                interp_per_z = RectBivariateSpline(logkMpc,mu,rsd_grid[:,:,iz])
                rsd_ap[:,:,iz] = interp_per_z(np.log(k_ap[:,:,iz].to(u.Mpc**-1).value), mu_ap[:,:,iz], grid=False)
            rsd_ap *= uI
        else:
            if self.astro.astroparams["v_of_M"]:
                # In this case the mean temperature is a function of k*mu due to the smoothing
                bterm_grid = np.reshape(self.bias_term(z, k=k, mu=mu, BAOpars=self.BAOpars),(*k.shape, *mu.shape, *z.shape))
                uI = bterm_grid.unit
                for iz, zi in enumerate(z):
                    interp_per_z = RectBivariateSpline(logkMpc, mu, bterm_grid[:,:,iz])
                    rsd_ap[:,:,iz] = (
                        interp_per_z(np.log(k_ap[:,:,iz].to(u.Mpc**-1).value),
                        mu_ap[:,:,iz],
                        grid=False)**2
                    )
                rsd_ap *= uI
            elif "f_NL" in self.cosmology.fullcosmoparams:
                # In this case the bias is scale dependent
                bterm_grid = np.reshape(self.bias_term(z, k=k, BAOpars=self.BAOpars),(*k.shape, *z.shape))
                uI = bterm_grid.unit
                for iz, zi in enumerate(z):
                    interp_per_z = UnivariateSpline(logkMpc,bterm_grid[:,iz])
                    rsd_ap[:,:,iz] = interp_per_z(np.log(k_ap[:,:,iz].to(u.Mpc**-1).value))**2
            else:
                rsd_ap = np.atleast_1d(self.bias_term(z, BAOpars=self.BAOpars)**2)[None,None,:]

        if cfg.settings["verbosity"] >1:
            trsd = time()
            print("Redshift space distortions obtained in {} seconds".format(trsd-tPk))

        # Obtain shotnoise contribution (AP effect only enters when computing scale dependent shot noise)
        if self.astro.astroparams["v_of_M"]:
            # The scale dependent supression due to v_of_M enters with k and mu
            Ps_ap = np.zeros(outputshape)
            Ps_grid = np.reshape(self.shotnoise(z, k=k, mu=mu, BAOpars=self.BAOpars),(*k.shape, *mu.shape, *z.shape))
            logPs_grid = np.log(Ps_grid.value)
            uI = Ps_grid.unit
            for iz, zi in enumerate(z):
                interp_per_z = RectBivariateSpline(logkMpc, mu, logPs_grid[:,:,iz])
                Ps_ap[:,:,iz] = np.exp(
                    interp_per_z(np.log(k_ap[:,:,iz].to(u.Mpc**-1).value),
                    mu_ap[:,:,iz],
                    grid=False))
            Ps_ap *= uI

        elif cfg.settings["halo_model_PS"]:
            # In this case, the one-halo self correlation is only a function of k
            Ps_ap = np.zeros(outputshape)
            Ps_grid = np.reshape(self.shotnoise(z, k=k, BAOpars=self.BAOpars),(*k.shape,*z.shape))
            logPs_grid = np.log(Ps_grid.value)
            uI = Ps_grid.unit
            for iz, zi in enumerate(z):
                interp_per_z = UnivariateSpline(logkMpc,logPs_grid[:,iz])
                Ps_ap[:,:,iz] = np.exp(interp_per_z(np.log(k_ap[:,:,iz].to(u.Mpc**-1).value)))
            Ps_ap *= uI

        else:
            Ps_ap = np.atleast_1d(self.shotnoise(z, BAOpars=self.BAOpars))[None,None,:]

        if cfg.settings["verbosity"] >1:
            tps = time()
            print("Shot-noise obtained in {} seconds".format(tps-trsd))

        self.Pk_Obs = (
            (qparr * np.power(qperp,2))[None,None,:]
            * (rsd_ap * Pk_ap + Ps_ap)
            * Fnu
        )

        if cfg.settings["Smooth_window"]:
            self.Pk_true = copy(self.Pk_Obs)
            self.Pk_Obs = self.convolved_Pk()

        if cfg.settings["verbosity"] >1:
            print("Observed power spectrum obtained in {} seconds".format(time()-tstart))

    def compute_power_spectra_moments(self):
        """
        This function computes the power spectrum monopole, quadropole and hexadecapole
        For other moments it creats a new callable function to compute them
        """
        #grid quantities
        mu = self.mu
        Pobs = self.Pk_Obs

        def Pk_ell_moments(ell):
            norm = (2*ell+1)/2
            L_ell = legendre(ell)
            return simpson(y=Pobs * norm * L_ell(mu)[None,:,None],x=mu,axis=1) * Pobs.unit

        self.Pk_0bs = Pk_ell_moments(0)
        self.Pk_2bs = Pk_ell_moments(2)
        self.Pk_4bs = Pk_ell_moments(4)
        self.Pk_ell_moments = Pk_ell_moments

    def create_pk_interp(self):
        k = self.k
        mu = self.mu
        z = self.z
        Pk_Obs = self.Pk_Obs

        # fix units go to logscale
        uP = Pk_Obs.unit
        logP = np.log(Pk_Obs.value)
        uk = k.unit
        logk = np.log(k.value)

        # interpolate in loglog
        interp_per_z = []
        for iz,zi in enumerate(z):
            interp_per_z.append(RectBivariateSpline(logk,mu,logP[:,:,iz]))

        def Pk_from_grid(pk,pmu):
            pk = np.atleast_1d(pk)
            pmu = np.atleast_1d(pmu)
            logpk = np.log(pk.to(uk).value)
            fromgrid = np.array([interp(logpk,pmu) for interp in interp_per_z])
            result = np.exp(fromgrid) * uP
            return np.squeeze(result)

        return Pk_from_grid
