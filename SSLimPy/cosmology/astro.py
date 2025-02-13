import astropy.units as u
import astropy.constants as c
import numpy as np

from copy import deepcopy
from scipy.interpolate import RectBivariateSpline

from SSLimPy.cosmology.halomodel import halomodel
from SSLimPy.cosmology.fitting_functions import luminosity_functions as lf
from SSLimPy.cosmology.fitting_functions import mass_luminosity as ml
from SSLimPy.interface import config as cfg
from SSLimPy.utils.utils import *


class astro_functions:
    def __init__(
        self, Halomodel: halomodel, astropars:dict=dict(),
    ):
        self.halomodel = Halomodel
        self.cosmology = Halomodel.cosmology

        # Units
        self.hubble = Halomodel.hubble
        self.Mpch = Halomodel.Mpch
        self.Msunh = Halomodel.Msunh

        self.astroparams = deepcopy(astropars)
        self._set_astrophysics_defaults()
        self.model_type = self.astroparams["model_type"]
        self.model_name = self.astroparams["model_name"]
        self.model_par = self.astroparams["model_par"]
        self.nu = self.astroparams["nu"]
        self.sigma_scater = self.astroparams["sigma_scatter"]
        self.fduty = self.astroparams["fduty"]

        ### TEXT VOMIT ###
        if cfg.settings["verbosity"] > 1:
            self.recap_astro()
        ##################

        # Internal samples for computations
        self.M = self.halomodel.M
        self.L = np.geomspace(
            self.astroparams["Lmin"], self.astroparams["Lmax"], self.astroparams["nL"]
        )

        # Check passed models
        self._init_model()

        # mass luminosity function
        self.massluminosityfunction = self._create_mass_luminosity()

        # halo luminosity function
        self.haloluminosityfunction = self._create_luminosty_function()

    def _set_astrophysics_defaults(self):
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
        self.astroparams.setdefault("Lmin", 10 * u.Lsun)
        self.astroparams.setdefault("Lmax", 1e8 * u.Lsun)
        self.astroparams.setdefault("nL", 5000)
        self.astroparams.setdefault("sigma_scatter", 0)
        self.astroparams.setdefault("fduty", 1)
        self.astroparams.setdefault("nu", cfg.obspars["nu"])

    def _init_model(self):
        """
        Check if model given by model_name exists in the given model_type
        """
        model_type = self.model_type
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

    def _create_mass_luminosity(self):
        if "ML" in self.model_type:
            L_of_M = getattr(
                self._mass_luminosity_function, self.astroparams["model_name"]
            )

        elif "LF" in self.model_type:
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

    def _create_luminosty_function(self):
        M = self.M
        z = self.cosmology.results.zgrid
        L = self.L

        if "ML" in self.model_type:
            sigma = np.maximum(cfg.settings["sigma_scatter"], 0.05)
            # Special case for Tony Li model- scatter does not preserve LCO
            if self.astroparams["model_name"] == "TonyLi":
                alpha = self.model_par["alpha"]
                sig_SFR = self.model_par["sig_SFR"]
                # assume sigma and sig_SFR are totally uncorrelated
                sigma = (sigma**2 + sig_SFR**2 / alpha**2) ** 0.5
            sigma_base_e = sigma * 2.302585

            dn_dM_of_M_and_z = np.reshape(
                self.halomodel.halomassfunction(M, z), (*M.shape, *z.shape)
            )
            L_of_M = np.reshape(self.massluminosityfunction(M, z), (*M.shape, *z.shape))

            flognorm = lognormal(
                L[None, :, None],
                np.log(L_of_M.value)[:, None, :] - 0.5 * sigma_base_e**2.0,
                sigma_base_e,
            )

            CFL = flognorm * dn_dM_of_M_and_z[:, None, :]
            dn_dL_of_L = np.trapz(CFL, self.M, axis=0)

        elif "LF" in self.model_type:
            dn_dL_of_L_func = getattr(
                self._luminosicosmology.results.zgridty_function, self.astroparams["model_name"]
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

    ###################
    # Astro Functions #
    ###################

    def nbar(self, z):
        """
        Mean number density of galaxies, computed from the luminosity function
        in 'LF' models and from the mass function in 'ML' models
        """
        model_type = self.model_type
        if model_type == "LF":
            dndL = self.haloluminosityfunction(self.L, z)
            nbar = np.trapz(dndL, self.L, axis=0)
        else:
            dndM = self.halomodel.halomassfunction(self.M, z)
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

    def Lavg(self, z, p=1):
        z = np.atleast_1d(z)
        if "ML" in self.model_type:
            log10 = np.log(10)
            M = self.M
            Lp = np.reshape(self.massluminosityfunction(M, z), (*M.shape, *z.shape))**p
            dndM = np.reshape(self.halomodel.halomassfunction(M, z), (*M.shape, *z.shape))
            Lpbar = np.trapz(M[:,None] * Lp * dndM, np.log(M.value)).to(u.Lsun)

            # Add L scatter
            Lpbar *= np.exp(0.5 * p * (p - 1) * (self.sigma_scater * log10)**2)*self.fduty

            if "TonyLi" == self.model_name:
                # LCO is nolonger conserved
                Lpbar *= np.exp(-0.5 * p * (p - 1) * (self.sigma_scater * log10)**2)

                alpha = self.model_par['alpha']
                sig_SFR = self.model_par['sig_SFR']
                # SFR scatter
                Lpbar *= np.exp(0.5 * p * (p - 1) * (1 / alpha * sig_SFR * log10)**2)
                # LCO scatter
                Lpbar *= np.exp(0.5 * p**2 * (self.sigma_scater * log10)**2)


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

        bh = restore_shape(self.halomodel.halobias(M, z, k=k), k, M, z)
        U = np.reshape(self.halomodel.ft_NFW(k, M, z), (*k.shape, *M.shape, *z.shape))
        LofM = np.reshape(self.massluminosityfunction(M, z), (*M.shape, *z.shape))
        dndM = np.reshape(self.halomodel.halomassfunction(M, z), (*M.shape, *z.shape))

        Fv = np.ones((*k.shape, *mu.shape, *M.shape, *z.shape))
        if self.astroparams["v_of_M"]:
            Fv = np.reshape(
                self.halomodel.broadening_FT(k, mu, M, z),
                (*k.shape, *mu.shape, *M.shape, *z.shape),
            )

        itgrnd1 = (
            Fv * LofM[None, None, :, :] * dndM[None, None, :, :] * (bh * U)[:, None, :, :]
        )
        itgrnd2 = (
            Fv * LofM[None, None, :, :] * dndM[None, None, :, :] * U[:, None, :, :]
        )

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
        dndM = self.halomodel._bias_function.sc_hmf(M, z, dc=dc)
        b = np.reshape(
            getattr(self.halomodel._bias_function, bstring)(M, z, dc=dc), (*M.shape, *z.shape)
        )

        itgrnd1 = M[:, None] * LofM**power * dndM * b
        itgrnd2 = M[:, None] * LofM**power * dndM

        I1 = np.trapz(itgrnd1, np.log(M.value), axis=0)
        I2 = np.trapz(itgrnd2, np.log(M.value), axis=0)
        avgbL = I1 / I2
        return avgbL.to(1).value

    def Lmoments(self, z, k=None, mu=None, moment=1):
        """
        Sky-averaged luminosity density moments at nuObs from target line.
        Has two cases for 'LF' and 'ML' models that where handled in create_luminosty_function
        """
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)
        model_type = self.model_type
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
            dndM = np.reshape(self.halomodel.halomassfunction(M, z), (1, 1, *M.shape, *z.shape))
            L = np.reshape(
                self.massluminosityfunction(M, z), (1, 1, *M.shape, *z.shape)
            )

            Fv = np.ones((*k.shape, *mu.shape, *M.shape, *z.shape))
            if self.astroparams["v_of_M"]:
                Fv = np.reshape(
                    self.halomodel.broadening_FT(k, mu, M, z),
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

    def recap_astro(self):
        print("Astronomical Parameters:")
        for key in self.astroparams:
            print("   " + key + ": {}".format(self.astroparams[key]))
