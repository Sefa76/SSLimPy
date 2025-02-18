import astropy.units as u
import astropy.constants as c
import numpy as np

from copy import deepcopy
from scipy.interpolate import RectBivariateSpline

from SSLimPy.cosmology.halo_model import HaloModel
from SSLimPy.cosmology.fitting_functions import luminosity_functions as lf
from SSLimPy.cosmology.fitting_functions import mass_luminosity as ml
from SSLimPy.interface.survey_specs import SurveySpecifications
from SSLimPy.interface import config as cfg
from SSLimPy.utils.utils import *


class AstroFunctions:
    def __init__(
        self,
        halomodel: HaloModel,
        survey_specs: SurveySpecifications,
        astropars: dict = dict(),
    ):
        self.halomodel = halomodel
        self.cosmology = halomodel.cosmology
        self.survey_specs = survey_specs

        # Units
        self.hubble = halomodel.hubble
        self.Mpch = halomodel.Mpch
        self.Msunh = halomodel.Msunh

        self.astroparams = deepcopy(astropars)
        self._set_astrophysics_defaults()
        self.model_type = self.astroparams["model_type"]
        self.model_name = self.astroparams["model_name"]
        self.model_par = self.astroparams["model_par"]
        self.nu = self.survey_specs.obsparams["nu"]
        self.nuObs = self.survey_specs.obsparams["nuObs"]
        self.sigma_scatter = self.astroparams["sigma_scatter"]
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

    def _init_model(self):
        """
        Check if model given by model_name exists in the given model_type
        """
        model_type = self.model_type
        model_name = self.astroparams["model_name"]
        self._luminosity_function = lf.luminosity_functions(self, self.model_par)
        self._mass_luminosity_function = ml.mass_luminosity(self, self.model_par)

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
                "Mcut_min": self.halomodel.Mmin,
                "Mcut_max": self.halomodel.Mmax,
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
            sigma = np.maximum(self.sigma_scatter, 0.05)
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
        """Mean halo luminosity and higher powers."""
        z = np.atleast_1d(z)
        if "ML" in self.model_type:
            log10 = np.log(10)
            M = self.M
            Lp = (
                np.reshape(self.massluminosityfunction(M, z), (*M.shape, *z.shape)) ** p
            )
            dndM = np.reshape(
                self.halomodel.halomassfunction(M, z), (*M.shape, *z.shape)
            )
            Lpbar = np.trapz(M[:, None] * Lp * dndM, np.log(M.value), axis=0).to(u.Lsun)

            # Add L scatter
            Lpbar *= np.exp(0.5 * p * (p - 1) * (self.sigma_scatter * log10) ** 2)
            if "TonyLi" == self.model_name:
                # LCO is nolonger conserved
                Lpbar *= np.exp(0.5 * p * (self.sigma_scatter * log10) ** 2)

                alpha = self.model_par["alpha"]
                sig_SFR = self.model_par["sig_SFR"]
                # SFR scatter
                Lpbar *= np.exp(0.5 * p * (p - 1) * (sig_SFR / alpha * log10) ** 2)
        else:
            L = self.L
            haloluminosity = np.reshape(
                self.haloluminosityfunction(L, z),
                (*L.shape, *z.shape),
            )
            Lpbar = np.trapz(
                L[:, None] ** (p + 1) * haloluminosity, np.log(L.value), axis=0
            )

        return Lpbar * self.fduty

    def Tavg(self, z, p=1):
        return self.CLT(z) ** p * self.Lavg(z, p=p)

    def bavg(self, bstring, z, power, dc=1.6865):
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
            getattr(self.halomodel._bias_function, bstring)(M, z, dc=dc),
            (*M.shape, *z.shape),
        )

        itgrnd1 = M[:, None] * LofM**power * dndM * b
        itgrnd2 = M[:, None] * LofM**power * dndM

        I1 = np.trapz(itgrnd1, np.log(M.value), axis=0)
        I2 = np.trapz(itgrnd2, np.log(M.value), axis=0)
        avgbL = I1 / I2
        return avgbL.to(1).value

    ##################
    # Halo integrals #
    ##################

    def Lhalo(self, z, *args, p=1):
        """Luminosity weight higher order halo profiles for n-halo terms
        Computes the mean halo profile weight with some power of the luminosity.
        this shows up for example in the halo shot noise,
        or when appoximating that the mean of higher order biases are
        independent from the scale dependence of the halo profile
        """
        M = self.M.to(u.Msun)
        z = np.atleast_1d(z)

        if len(args) % p != 0:
            raise ValueError("You have to pass wave-vectors for every p")
        else:
            kd = [np.atleast_1d(args[ik]) for ik in range(p)]

        # Independent of k
        L_of_M = np.reshape(self.massluminosityfunction(M, z), (*M.shape, *z.shape))
        dndM = np.reshape(self.halomodel.halomassfunction(M, z), (*M.shape, *z.shape))

        # Dependent on k
        normhaloprofile = []
        for ik in range(p):
            k = kd[ik]
            U = np.reshape(
                self.halomodel.ft_NFW(k, M, z), (*k.shape, *M.shape, *z.shape)
            )
            U = np.expand_dims(U, (*range(ik), *range(ik + 1, p)))
            U = np.expand_dims(U, (*range(p, 2 * p),))
            normhaloprofile.append(U)

        # Dependent on k and mu
        Fv = np.ones((p,))
        if self.halomodel.haloparams["v_of_M"]:
            Fv = []
            for ik in range(p):
                k = kd[ik]
                mu = np.atleast_1d(args[p + ik])
                F = np.reshape(
                    self.halomodel.broadening_FT(k, mu, M, z),
                    (*k.shape, *mu.shape, *M.shape, *z.shape),
                )
                F = np.expand_dims(
                    F,
                    (*range(ik), *range(ik + 1, p)),
                )
                F = np.expand_dims(
                    F,
                    (*range(p, p + ik), *range(p + ik + 1, 2 * p)),
                )
                Fv.append(F)

        # Construct the integrand
        I1 = dndM * L_of_M**p * M[:, None]
        I2 = I1
        for ik in range(p):
            I2 = I2 * Fv[ik] * normhaloprofile[ik]
        logM = np.log(M.value)
        Umean = (np.trapz(I2, logM, axis=-2) / np.trapz(I1, logM, axis=-2)).to(1).value
        return np.squeeze(self.Lavg(z, p=p) * Umean)

    def Thalo(self, z, *args, p=1):
        return self.CLT(z) ** p * self.Lhalo(z, *args, p=p)

    def T_one_halo(self, k, z, mu=None):
        """Directly computes the one-halo power spectrum."""
        M = self.M.to(u.Msun)
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)

        dndM = np.reshape(self.halomodel.halomassfunction(M, z), (*M.shape, *z.shape))
        L_of_M = np.reshape(
            self.massluminosityfunction(M, z) ** 2, (*M.shape, *z.shape)
        )
        U2 = np.reshape(
            self.halomodel.ft_NFW(k, M, z) ** 2, (*k.shape, *M.shape, *z.shape)
        )
        Fv = 1
        if self.halomodel.haloparams["v_of_M"]:
            Fv = np.reshape(
                self.halomodel.broadening_FT(k, mu, M, z) ** 2,
                (*k.shape, *mu.shape, *M.shape, *z.shape),
            )
        I1 = M[None, None, :, None] * L_of_M[None, None, :, :] * dndM[None, None, :, :]
        I2 = I1 * U2[:, None, :, :] * Fv
        Uavg = (
            (
                np.trapz(I2, np.log(M.value), axis=-2)
                / np.trapz(I1, np.log(M.value), axis=-2)
            )
            .to(1)
            .value
        )
        return np.squeeze(self.Tavg(z, p=2) * Uavg)

    def bhalo(self, k, z, mu=None):
        """Mean Tb, factor in front of the clutstering part of the LIM-autopower spectrum"""
        k = np.atleast_1d(k)
        z = np.atleast_1d(z)
        mu = np.atleast_1d(mu)
        M = self.M

        dndM = np.reshape(self.halomodel.halomassfunction(M, z), (*M.shape, *z.shape))
        L = np.reshape(self.massluminosityfunction(M, z), (*M.shape, *z.shape))
        b = restore_shape(self.halomodel.halobias(M, z, k=k), k, M, z)
        U = restore_shape(self.halomodel.ft_NFW(k, M, z), k, M, z)

        Fv = 1
        if self.halomodel.haloparams["v_of_M"]:
            Fv = np.reshape(
                self.halomodel.broadening_FT(k, mu, M, z),
                (*k.shape, *mu.shape, *M.shape, *z.shape),
            )

        I1 = L * dndM * M[:, None]
        I2 = I1 * Fv * U[:, None, :, :] * b[:, None, :, :]
        bmean = (
            (np.trapz(I2, np.log(M.value), axis=-2) / np.trapz(I1, np.log(M.value), axis=-2)).to(1).value
        )

        return np.squeeze(self.Tavg(z, p=1) * bmean)

    def recap_astro(self):
        print("Astronomical Parameters:")
        for key in self.astroparams:
            print("   " + key + ": {}".format(self.astroparams[key]))
