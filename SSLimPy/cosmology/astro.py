from copy import deepcopy
import astropy.constants as c
import astropy.units as u
import numpy as np
from scipy.interpolate import RectBivariateSpline
from SSLimPy.cosmology.cosmology import cosmo_functions
from SSLimPy.cosmology.fitting_functions import bias_fitting_functions as bf
from SSLimPy.cosmology.fitting_functions import coevolution_bias as cb
from SSLimPy.cosmology.fitting_functions import halo_mass_functions as HMF
from SSLimPy.cosmology.fitting_functions import luminosity_functions as lf
from SSLimPy.cosmology.fitting_functions import mass_luminosity as ml
from SSLimPy.interface import config as cfg
from SSLimPy.interface import updater
from SSLimPy.utils.utils import smooth_dW, smooth_W


class astro_functions:
    def __init__(
        self, cosmopars=dict(), astropars=dict(), cosmology: cosmo_functions = None
    ):

        self.astroparams = deepcopy(astropars)
        self.set_astrophysics_defaults()
        self.astrotracer = cfg.settings["astro_tracer"]

        ### TEXT VOMIT ###
        if cfg.settings["verbosity"] > 1:
            self.recap_astro()
        ##################
        if cosmology:
            self.cosmopars = cosmology.fullcosmoparams
            self.cosmology = cosmology
        else:
            self.cosmopars = cosmopars
            self.cosmology = updater.update_cosmo(cfg.fiducialcosmo, cosmopars)

        # Current units
        self.hubble = self.cosmology.h()
        self.Mpch = u.Mpc / self.hubble
        self.Msunh = u.Msun / self.hubble
        self.rho_crit = 2.77536627e11 * (self.Msunh * self.Mpch**-3).to(
            u.Msun * u.Mpc**-3
        )  # Msun/Mpc^3

        # Internal samples for computations
        self.M = np.geomspace(
            self.astroparams["Mmin"], self.astroparams["Mmax"], self.astroparams["nM"]
        )
        self.L = np.geomspace(
            self.astroparams["Lmin"], self.astroparams["Lmax"], self.astroparams["nL"]
        )
        # find the redshifts for frequencies asked for:
        self.nu = cfg.obspars["nu"]
        self.nuObs = cfg.obspars["nuObs"]

        self.sigmaM, self.dsigmaM_dM = self.create_sigmaM_funcs(self.astro_tracer)

        # Check passed models
        self.init_model()
        self.init_halo_mass_function()
        self.init_bias_function()

        # bias function
        # !Without Corrections for non-Gaussianity!
        self.delta_crit = 1.686
        self._b_of_M = getattr(self._bias_function, self.astroparams["bias_model"])
        # use halo-bias to compute the bias with all corrections

        # halo mass function
        # !Without Corrections for non-Gaussianity!
        self._dn_dM_of_M = getattr(
            self._halo_mass_function, self.astroparams["hmf_model"]
        )
        # use halomassfunction to compute the bias with all corrections

        # mass luminosity function
        self.massluminosityfunction = self.create_mass_luminosity()

        # halo luminosity function
        self.haloluminosityfunction = self.create_luminosty_function()

        # higher order bias computations
        self.bias_coevolution = cb.coevolution_bias(self)

    ###################
    # Astro Functions #
    ###################

    def mass_non_linear(self, z, delta_crit=1.686):
        """
        Get (roughly) the mass corresponding to the nonlinear scale in units of Msun h
        """
        sigmaM_z = self.sigmaM(self.M, z)
        mass_non_linear = self.M[np.argmin(np.power(sigmaM_z - delta_crit, 2), axis=0)]

        return mass_non_linear.to(self.Msunh)

    def halobias(self, M, z, k=None):
        """
        This function is a wrapper to obtain the halo bias with all correction and as a function of standard inputs
        """
        k = np.atleast_1d(k)
        M = np.atleast_1d(M)
        z = np.atleast_1d(z)

        bh = self._b_of_M(M, z, self.delta_crit)

        if "f_NL" in self.cosmology.fullcosmoparams:
            Delta_b = np.reshape(self.Delta_b(k, M, z), (*k.shape, *M.shape, *z.shape))
            bh = bh[None, :, :] + Delta_b

        return np.squeeze(bh)

    def halomassfunction(self, M, z):
        """
        This function is a wrapper to obtain the halo mass function with all correction and as a function of standard inputs
        """
        M = np.atleast_1d(M).to(self.Msunh)
        z = np.atleast_1d(z)

        rho_input = (
            2.77536627e11
            * self.cosmology.Omega(0, self.astrotracer)
            * (self.Msunh * self.Mpch**-3)
        )

        dndM = np.reshape(self._dn_dM_of_M(M, rho_input, z), (*M.shape, *z.shape))

        if "f_NL" in self.cosmology.fullcosmoparams:
            Delta_HMF = np.reshape(self.Delta_HMF(M, z), (*M.shape, *z.shape))
            dndm *= 1 + Delta_HMF

        return np.squeeze(dndM).to(u.Mpc**-3 * u.Msun**-1)

    def sigmav_broadening(self, M, z):
        """Computes the physical scale of the line broadening due to
        galactic rotation curves.
        """
        Mvec = np.atleast_1d(M)[:, None]
        zvec = np.atleast_1d(z)[None, :]

        vM = np.atleast_1d(self.astroparams["v_of_M"](Mvec))
        Hz = np.atleast_1d(self.cosmology.Hubble(zvec))
        sv = vM / self.cosmology.celeritas * (1 + zvec) / Hz / np.sqrt(8 * np.log(2))

        return np.squeeze(sv)

    def broadening_FT(self, k, mu, M, z):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        M = np.atleast_1d(M)
        z = np.atleast_1d(z)

        sv = np.reshape(self.sigmav_broadening(M, z), (1, 1, *M.shape, *z.shape))
        fac = np.exp(
            -1 / 2 * np.power(k[:, None, None, None] * mu[None, :, None, None] * sv, 2)
        )
        return np.squeeze(fac)

    def bavg(self, z, k=None, mu=None):
        """
        Average luminosity-weighted bias for the given cosmology and line
        model. Assumed to be linearly weight

        Includes the effects of f_NL though the wrapping functions in astro
        """
        # Integrands for mass-averaging
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        M = self.M.to(self.Msunh)
        z = np.atleast_1d(z)

        LofM = np.reshape(self.massluminosityfunction(M, z), (*M.shape, *z.shape))
        dndM = np.reshape(self.halomassfunction(M, z), (*M.shape, *z.shape))
        bh = self.restore_shape(self.halobias(M, z, k=k), k, M, z)

        Fv = np.ones((*k.shape, *mu.shape, *M.shape, *z.shape))
        if self.astroparams["v_of_M"]:
            Fv = np.reshape(
                self.broadening_FT(k, mu, M, z),
                (*k.shape, *mu.shape, *M.shape, *z.shape),
            )

        itgrnd1 = (
            Fv * LofM[None, None, :, :] * dndM[None, None, :, :] * bh[:, None, :, :]
        )
        itgrnd2 = Fv * LofM[None, None, :, :] * dndM[None, None, :, :]

        I1 = np.trapz(itgrnd1, M, axis=2)
        I2 = np.trapz(itgrnd2, M, axis=2)
        b_line = I1 / I2
        return np.squeeze(b_line.to(1).value)

    def bavghalo(self, bstring, z, power, dc=1.6865):
        """
        Average luminosity-weighted bias for higher order bias functions

        Pass which bias you want as a string present in bias_coevolution
        Will default back to use ST fitting function as halo mass function!
        """
        # Integrands for mass-averaging
        M = self.M.to(self.Msunh)
        z = np.atleast_1d(z)

        LofM = np.reshape(self.massluminosityfunction(M, z), (*M.shape, *z.shape))
        dndM = self.bias_coevolution.sc_hmf(M, z, dc=dc)
        b = np.reshape(
            getattr(self.bias_coevolution, bstring)(M, z, dc=dc), (*M.shape, *z.shape)
        )

        itgrnd1 = M[:, None] * LofM**power * dndM * b
        itgrnd2 = M[:, None] * LofM**power * dndM

        I1 = np.trapz(itgrnd1, np.log(M.value), axis=0)
        I2 = np.trapz(itgrnd2, np.log(M.value), axis=0)
        avgbL = I1 / I2
        return avgbL.to(1).value

    def nbar(self, z):
        """
        Mean number density of galaxies, computed from the luminosity function
        in 'LF' models and from the mass function in 'ML' models
        """
        model_type = self.astroparams["model_type"]
        if model_type == "LF":
            dndL = self.haloluminosityfunction(self.L, z)
            nbar = np.trapz(dndL, self.L, axis=0)
        else:
            dndM = self.halomassfunction(self.M, z)
            nbar = np.trapz(dndM, self.M, axis=0)
        return nbar

    def CLT(self, z):
        if cfg.settings["do_Jysr"]:
            x = c.c / (
                4.0
                * np.pi
                * self.nu
                * self.cosmology.Hubble(z, physical=True)
                * (1.0 * u.sr)
            )
            CLT = x.to(u.Jy * u.Mpc**3 / (u.Lsun * u.sr))
        else:
            x = (
                c.c**3
                * (1 + z) ** 2
                / (
                    8
                    * np.pi
                    * c.k_B
                    * self.nu**3
                    * self.cosmology.Hubble(z, physical=True)
                )
            )
            CLT = x.to(u.uK * u.Mpc**3 / u.Lsun)
        return CLT

    def Lmoments(self, z, k=None, mu=None, moment=1):
        """
        Sky-averaged luminosity density moments at nuObs from target line.
        Has two cases for 'LF' and 'ML' models that where handled in create_luminosty_function
        """
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)
        model_type = self.astroparams["model_type"]
        if model_type == "LF":
            if self.astroparams["v_of_M"]:
                raise ValueError("Line width modelling only available for ML models")
            else:
                L = self.L
                dndL = np.reshape(
                    self.haloluminosityfunction(L, z), (*L.shape, *z.shape)
                )
                itgrnd1 = dndL * np.power(L, moment)[:, None]
                Lmoment = np.trapz(itgrnd1, L, axis=0)
        elif model_type == "ML":
            M = self.M
            dndM = np.reshape(self.halomassfunction(M, z), (1, 1, *M.shape, *z.shape))
            L = np.reshape(
                self.massluminosityfunction(M, z), (1, 1, *M.shape, *z.shape)
            )

            Fv = np.ones((*k.shape, *mu.shape, *M.shape, *z.shape))
            if self.astroparams["v_of_M"]:
                Fv = np.reshape(
                    self.broadening_FT(k, mu, M, z),
                    (*k.shape, *mu.shape, *M.shape, *z.shape),
                )

            itgrnd = dndM * np.power(L * Fv, moment)
            Lmoment = np.trapz(itgrnd, M, axis=2)

            # Special case for Tony Li model- scatter does not preserve LCO
            if self.astroparams["model_name"] == "TonyLi":
                model_pars = self.astroparams["model_par"]
                alpha = model_pars["alpha"]
                sig_SFR = model_pars["sig_SFR"]
                correction = np.exp(
                    (moment * alpha**-2 - alpha**-1)
                    * moment
                    * sig_SFR**2
                    * np.log(10) ** 2
                    / 2
                )
                Lmoment *= correction

        return np.squeeze(Lmoment)

    def Tmoments(self, z, k=None, mu=None, moment=1):
        """
        Sky-averaged brightness temperature moments at nuObs from target line.
        Else, you can directly input Tmean using TOY model
        """
        return np.power(self.CLT(z), moment) * self.Lmoments(
            z, k=k, mu=mu, moment=moment
        )

    #############################
    # Additional Init Functions #
    #############################

    def create_sigmaM_funcs(self, tracer="matter"):
        """
        This function creates the interpolating functions for sigmaM and dsigamM
        """
        M_inter = self.M
        z_inter = self.cosmology.results.zgrid

        sigmaM = self._sigmaM_of_z(M_inter, z_inter, tracer)
        dsigmaM = self._dsigmaM_of_z(M_inter, z_inter, tracer)

        # create interpolating functions
        logM_in_Msun = np.log(M_inter.to(u.Msun).value)
        logsigmaM = np.log(sigmaM)
        logmdsigmaM = np.log(-dsigmaM.to(u.Msun**-1).value)

        inter_logsigmaM = RectBivariateSpline(logM_in_Msun, z_inter, logsigmaM)
        inter_logmdsigmaM = RectBivariateSpline(logM_in_Msun, z_inter, logmdsigmaM)

        # restore units
        def sigmaM_of_M_and_z(M, z):
            M = np.atleast_1d(M)
            z = np.atleast_1d(z)
            logM = np.log(M.to(u.Msun).value)
            return np.squeeze(np.exp(inter_logsigmaM(logM, z)))

        def dsigmaM_of_M_and_z(M, z):
            M = np.atleast_1d(M.to(u.Msun))
            z = np.atleast_1d(z)
            logM = np.log(M.value)
            return np.squeeze(-np.exp(inter_logmdsigmaM(logM, z)) * u.Msun**-1)

        return sigmaM_of_M_and_z, dsigmaM_of_M_and_z

    def _sigmaM_of_z(self, M, z, tracer="matter"):
        """
        Mass (or CDM+baryon) variance at target redshift.
        Used to create interpolated versions
        """
        rhoM = self.rho_crit * self.cosmology.Omega(0, tracer)
        R = (3.0 * M / (4.0 * np.pi * rhoM)) ** (1.0 / 3.0)

        return self.cosmology.sigmaR_of_z(R, z, tracer)

    def _dsigmaM_of_z(self, M, z, tracer="matter"):
        """
        Matter (or CDM+baryon) derivative variance at target redshift.
        Used to create interpolated versions
        """
        rhoM = self.rho_crit * self.cosmology.Omega(0, tracer)
        R = (3.0 * M / (4.0 * np.pi * rhoM)) ** (1.0 / 3.0)

        return self.cosmology.dsigmaR_of_z(R, z, tracer) * (R / (3 * M))

    def init_model(self):
        """
        Check if model given by model_name exists in the given model_type
        """
        model_type = self.astroparams["model_type"]
        model_name = self.astroparams["model_name"]
        self._luminosity_function = lf.luminosity_functions(self)
        self._mass_luminosity_function = ml.mass_luminosity(self)

        if model_type == "ML" and not hasattr(
            self._mass_luminosity_function, model_name
        ):
            if hasattr(self._luminosity_function, model_name):
                raise ValueError(
                    model_name
                    + " not found in mass_luminosity.py."
                    + " Set model_type='LF' to use "
                    + model_name
                )
            else:
                raise ValueError(model_name + " not found in mass_luminosity.py")

        elif model_type == "LF" and not hasattr(self._luminosity_function, model_name):
            if hasattr(self._mass_luminosity_function, model_name):
                raise ValueError(
                    model_name
                    + " not found in luminosity_functions.py."
                    + " Set model_type='ML' to use "
                    + model_name
                )
            else:
                raise ValueError(model_name + " not found in luminosity_functions.py")

    def init_bias_function(self):
        """
        Initialise computation of bias function if model given by bias_model exists in the given model_type
        """
        bias_name = self.astroparams["bias_model"]
        self._bias_function = bf.bias_fittinig_functions(self)
        if not hasattr(self._bias_function, bias_name):
            raise ValueError(bias_name + " not found in bias_fitting_functions.py")

    def init_halo_mass_function(self):
        """
        Initialise computation of halo mass function if model given by hmf_model exists in the given model_type
        """
        hmf_model = self.astroparams["hmf_model"]
        self._halo_mass_function = HMF.halo_mass_functions(self)

        if not hasattr(self._halo_mass_function, hmf_model):
            raise ValueError(hmf_model + " not found in halo_mass_functions.py")

    def create_mass_luminosity(self):
        if "ML" in self.astroparams["model_type"]:
            L_of_M = getattr(
                self._mass_luminosity_function, self.astroparams["model_name"]
            )

        elif "LF" in self.astroparams["model_type"]:
            LF_par = {
                "A": 2.0e-6,
                "b": 1.0,
                "Mcut_min": self.astroparams["Mmin"],
                "Mcut_max": self.astroparams["Mmax"],
            }
            off_mass_luminosity = ml.mass_luminosity(self, LF_par)
            L_of_M = getattr(off_mass_luminosity, "MassPow")

        # The Mass Luminosity functions L(M,z) are already vectorized properly inside of the File
        return L_of_M

    def create_luminosty_function(self):
        M = self.M
        z = self.cosmology.results.zgrid
        L = self.L

        if "ML" in self.astroparams["model_type"]:
            sigma = np.maximum(cfg.settings["sigma_scatter"], 0.05)
            # Special case for Tony Li model- scatter does not preserve LCO
            if self.astroparams["model_name"] == "TonyLi":
                alpha = self.model_par["alpha"]
                sig_SFR = self.model_par["sig_SFR"]
                # assume sigma and sig_SFR are totally uncorrelated
                sigma = (sigma**2 + sig_SFR**2 / alpha**2) ** 0.5
            sigma_base_e = sigma * 2.302585

            dn_dM_of_M_and_z = np.reshape(
                self.halomassfunction(M, z), (*M.shape, *z.shape)
            )
            L_of_M = np.reshape(self.massluminosityfunction(M, z), (*M.shape, *z.shape))

            flognorm = self.lognormal(
                L[None, :, None],
                np.log(L_of_M.value)[:, None, :] - 0.5 * sigma_base_e**2.0,
                sigma_base_e,
            )

            CFL = flognorm * dn_dM_of_M_and_z[:, None, :]
            dn_dL_of_L = np.trapz(CFL, self.M, axis=0)

        elif "LF" in self.astroparams["model_type"]:
            dn_dL_of_L_func = getattr(
                self._luminosity_function, self.astroparams["model_name"]
            )

            dn_dL_of_L = dn_dL_of_L_func(L)[:, None] * np.ones_like(z)[None, :]

        logL = np.log(L.to(u.Lsun).value)
        dn_dL_of_L[dn_dL_of_L.value == 0] = 1e-99 * dn_dL_of_L.unit
        logLF = np.log(dn_dL_of_L.to(u.Mpc ** (-3) * u.Lsun ** (-1)).value)
        logLF_inter = RectBivariateSpline(logL, z, logLF)

        def HaloLuminosityFunction(pL, pz):
            pL = np.atleast_1d(pL.to(u.Lsun))
            pz = np.atleast_1d(pz)
            logpL = np.log(pL.value)
            logdndL = logLF_inter(logpL, pz)
            LF = u.Mpc ** (-3) * u.Lsun ** (-1) * np.exp(logdndL)
            return np.squeeze(LF)

        return HaloLuminosityFunction

    #########################################
    # f_NL corrections to HMF and halo bias #
    #########################################

    def S3_dS3(self, M, z):
        """
        The skewness and derivative with respect to mass of the skewness.
        Used to calculate the correction to the HMF due to non-zero fnl,
        as presented in 2009.01245.

        Their parameter k_cut is equivalent to our klim, not to be confused
        with the ncdm parameter. k_lim represents the cutoff in the skewness
        integral, we opt for no cutoff and thus set it to a very small value.
        This can be changed if necessary.
        """
        tracer = self.astrotracer
        M = np.atleast_1d(M)
        z = np.atleast_1d(z)

        rhoM = self.rho_crit * self.cosmology.Omega(0, tracer=tracer)
        R = (3.0 * M / (4.0 * np.pi * rhoM)) ** (1.0 / 3.0)

        kmin = self.cosmology.results.kmin_pk
        kmax = self.cosmology.results.kmax_pk
        k = np.geomspace(kmin, kmax, 128) / u.Mpc
        mu = np.linspace(-0.995, 0.995, 128)

        # Why Numpy is just the best

        #############################
        # Indicies k1, k2, mu, M, z #
        #############################

        # funnctions of k1 or k2 only
        P_phi = 9 / 25 * self.cosmology.primordial_scalar_pow(k)
        k_1 = k[:, None, None, None, None]
        k_2 = k[None, :, None, None, None]
        P_1 = P_phi[:, None, None, None, None]
        P_2 = P_phi[None, :, None, None, None]

        # functions of k1 or k2 and M
        tM = M[None, :]
        tR = R[None, :]
        tk = k[:, None]
        x = tR * tk
        j1 = np.reshape(smooth_W(x.flatten()), x.shape)
        dj1_dM = np.reshape(smooth_dW(x.flatten()), x.shape) * x / (3 * tM)
        j1_1 = j1[:, None, None, :, None]
        j1_2 = j1[None, :, None, :, None]
        dj1_dM_1 = dj1_dM[:, None, None, :, None]
        dj1_dM_2 = dj1_dM[None, :, None, :, None]

        # functions of k1 or k2 and z
        Tm = (
            -5
            / 3
            * np.reshape(
                self.cosmology.Transfer(k, z, nonlinear=False, tracer=tracer),
                (*k.shape, *z.shape),
            )
        )
        Tm_1 = Tm[:, None, None, None, :]
        Tm_2 = Tm[None, :, None, None, :]

        # functions of k1, k2, and mu
        tk_1 = k[:, None, None]
        tk_2 = k[None, :, None]
        tmu = mu[None, None, :]
        k_12 = np.sqrt(
            np.abs(np.power(tk_1, 2) + np.power(tk_2, 2) + 2 * tk_1 * tk_2 * tmu)
        )

        # functions of k1, k2, mu, and M
        tk = k_12.flatten()
        kmask = (tk > kmin) & (tk < kmax)
        xmasked = R[None, :] * tk[kmask, :]
        j1_12 = np.zeros((*tk.shape, *R.shape))
        dj1_dM_12 = np.zeros_like(j1_12)
        j1_12[kmask, :] = np.reshape(
            smooth_W(xmasked.flatten()), (*k.shape, *k.shape, *mu.shape, *M.shape)
        )
        dj1_dM_12[kmask, :] = (
            np.reshape(
                smooth_W(xmasked.flatten()), (*k.shape, *k.shape, *mu.shape, *M.shape)
            )
            * xmasked
            / (3 * tM)
        )
        j1_12 = j1_12[:, :, :, :, None]
        dj1_dM_12 = dj1_dM_12[:, :, :, :, None]

        # functions of k1, k2, mu, and z
        Tm_12 = np.zeros((*tk.shape, *R.shape))
        Tm_12[kmask, :] = (
            -5
            / 3
            * np.reshape(
                self.cosmology.Transfer(tk[kmask], z, nonlinear=False, tracer=tracer),
                (*k.shape, *k.shape, *mu.shape, *z.shape),
            )
        )
        Tm_12 = Tm_12[:, :, :, None, :]

        # Integrandts
        W = j1_1 * j1_2 * j1_12
        dW_dM = (
            j1_1 * j1_2 * dj1_dM_12 + j1_1 * dj1_dM_2 * j1_12 + dj1_dM_1 * j1_2 * j1_12
        )

        integ_S3 = (
            np.power(k_1, 2) * np.power(k_2, 2) * W * Tm_1 * Tm_2 * Tm_12 * P_1 * P_2
        )
        integ_dS3_dM = (
            np.power(k_1, 2)
            * np.power(k_2, 2)
            * dW_dM
            * Tm_1
            * Tm_2
            * Tm_12
            * P_1
            * P_2
        )

        # The integration
        S3 = np.trapz(integ_S3, k, axis=0)
        S3 = np.trapz(S3, k, axis=0)
        S3 = np.trapz(S3, mu, axis=0)
        dS3_dM = np.trapz(integ_dS3_dM, k, axis=0)
        dS3_dM = np.trapz(dS3_dM, k, axis=0)
        dS3_dM = np.trapz(dS3_dM, mu, axis=0)

        fac = self.cosmology.fullcosmoparams["f_NL"] * 6 / 8 / np.pi**4
        S3 = -1 * fac * np.squeeze(S3)
        dS3_dm = -1 * fac * np.squeeze(dS3_dm)
        return -S3, dS3_dm

    def kappa3_dkappa3(self, M, z):
        """
        Calculates kappa_3 its derivative with respect to halo mass M from 2009.01245
        """
        S3, dS3_dM = self.S3_dS3(M, z, self.astrotracer)
        sigmaM = self.sigmaM(M, z)
        dSigmaM = self.dsigmaM_dM(M, z, self.astrotracer)

        kappa3 = S3 / sigmaM
        dkappa3dM = (dS3_dM - 3 * S3 * dSigmaM / sigmaM) / (sigmaM**3)

        return kappa3, dkappa3dM

    def Delta_HMF(self, M, z):
        """
        The correction to the HMF due to non-zero f_NL, as presented in 2009.01245.
        """
        sigmaM = self.sigmaM(M, z)
        dSigmaM = self.dsigmaM_dM(M, z)

        nuc = 1.42 / sigmaM
        dnuc_dM = -1.42 * dSigmaM / (sigmaM) ** 2

        kappa3, dkappa3_dM = self.kappa3_dkappa3(M, z)

        H2nuc = nuc**2 - 1
        H3nuc = nuc**3 - 3 * nuc

        F1pF0p = (kappa3 * H3nuc - H2nuc * dkappa3_dM / dnuc_dM) / 6

        return F1pF0p

    def Delta_b(self, k, M, z):
        """
        Scale dependent correction to the halo bias in presence of primordial non-gaussianity
        """
        tracer = self.astrotracer

        M = np.atleast_1d(M)
        z = np.atleast_1d(z)
        k = np.atleast_1d(k)

        Tk = np.reshape(self.cosmology.Transfer(k, z), (*k.shape, *z.shape))
        bias = np.reshape(self._b_of_M(M, z, self.delta_crit), (*M.shape, *z.shape))

        f1 = (self.cosmology.Hubble(0, physical=True) / (c.c * k)).to(1).value
        f2 = (
            3
            * self.cosmology.Omega(0, tracer=tracer)
            * self.cosmology.fullcosmoparams["f_NL"]
        )

        f1_of_k = f1[:, None, None]
        Tk_of_k_and_z = Tk[:, None, :]
        bias_of_M_and_z = bias[None, :, :]

        # Compute non-Gaussian correction Delta_b
        delta_b = (bias_of_M_and_z - 1) * f2 * f1_of_k / Tk_of_k_and_z
        return np.squeeze(delta_b)

    ##################
    # Helper Functions
    ##################
    # To be Outsorced
    def lognormal(self, x, mu, sigma):
        """
        Returns a lognormal PDF as function of x with mu and sigma
        being the mean of log(x) and standard deviation of log(x), respectively
        """
        try:
            return (
                1
                / x
                / sigma
                / (2.0 * np.pi) ** 0.5
                * np.exp(-((np.log(x.value) - mu) ** 2) / 2.0 / sigma**2)
            )
        except:
            return (
                1
                / x
                / sigma
                / (2.0 * np.pi) ** 0.5
                * np.exp(-((np.log(x) - mu) ** 2) / 2.0 / sigma**2)
            )

    def restore_shape(self, A, *args):
        """
        Extremely dangerous function to reshape squeezed arrays into arrays with boradcastable shapes
        Only use when there is other way as this assumes that the output shape has lenghs corresponding to input
        And is sqeezed in order of the input
        """
        A = np.atleast_1d(A)
        inputShape = A.shape
        targetShape = ()
        for arg in args:
            targetShape = (*targetShape, *np.atleast_1d(arg).shape)

        inputShape = np.array(inputShape)
        targetShape = np.array(targetShape)

        new_shape_A = []
        j = 0
        for i in range(len(targetShape)):
            if j < len(inputShape) and inputShape[j] == targetShape[i]:
                new_shape_A.append(inputShape[j])
                j += 1
            else:
                new_shape_A.append(1)

        A = A.reshape(new_shape_A)
        return A

    def set_astrophysics_defaults(self):
        """
        Fills up default values in the astropars dictionary if the values are not found.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        self.astroparams.setdefault("model_type", "LF")
        self.astroparams.setdefault("model_name", "SchCut")
        self.astroparams.setdefault(
            "model_par",
            {
                "phistar": 9.6e-11 * u.Lsun**-1 * u.Mpc**-3,
                "Lstar": 2.1e6 * u.Lsun,
                "alpha": -1.87,
                "Lmin": 5000 * u.Lsun,
            },
        )
        self.astroparams.setdefault("hmf_model", "ST")
        self.astroparams.setdefault("bias_model", "ST99")
        self.astroparams.setdefault("bias_par", {})
        self.astroparams.setdefault("Mmin", 1e9 * u.Msun)
        self.astroparams.setdefault("Mmax", 1e15 * u.Msun)
        self.astroparams.setdefault("nM", 500)
        self.astroparams.setdefault("Lmin", 10 * u.Lsun)
        self.astroparams.setdefault("Lmax", 1e8 * u.Lsun)
        self.astroparams.setdefault("nL", 5000)
        self.astroparams.setdefault("v_of_M", None)
        self.astroparams.setdefault("line_incli", True)

    def recap_astro(self):
        print("Astronomical Parameters:")
        for key in self.astroparams:
            print("   " + key + ": {}".format(self.astroparams[key]))
