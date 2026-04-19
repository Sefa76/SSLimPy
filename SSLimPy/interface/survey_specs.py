"""
This module was desinged to learn about PEP typing and interfaces with linters / mypy.
The rest of the code does not need to follow this style.
"""

from abc import ABC, abstractmethod
from copy import copy
from typing import Protocol, Union

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from scipy.special import spherical_jn, jn

from SSLimPy.interface.config import Configuration
from SSLimPy.cosmology.cosmology import CosmoFunctions
from SSLimPy.utils.utils import *


class SurveyInterface(Protocol):
    """This protocol ensures the correct ducktypes for the SurveySpecifications class"""

    cosmology: CosmoFunctions
    obsparams: dict

    def __init__(self, obspars: dict, cosmo: CosmoFunctions) -> None: ...
    def set_survey_defaults(self) -> None: ...
    def get_redshifts(self) -> tuple: ...
    def F_parr(
        self, k: Quantity, mu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]: ...
    def F_perp(
        self, k: Quantity, mu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]: ...
    def detector_noise(self) -> Quantity: ...


class SurveySpecifications(ABC):
    """The Survey Specifications for different types of surveys
    (Galaxies, LIM, etc) should be direct extensions to this baseclass,
    and implenent all its methods"""

    @abstractmethod
    def __init__(self, obspars: dict, cosmo: CosmoFunctions) -> None: ...

    @property
    @abstractmethod
    def cosmology(self) -> CosmoFunctions: ...

    @cosmology.setter
    @abstractmethod
    def cosmology(self, pcosmology: CosmoFunctions) -> None: ...

    @property
    @abstractmethod
    def obsparams(self) -> dict: ...

    @obsparams.setter
    @abstractmethod
    def obsparams(self, pobsparams: dict) -> None: ...

    @abstractmethod
    def set_survey_defaults(self) -> None: ...

    @abstractmethod
    def get_redshifts(self) -> tuple: ...

    @abstractmethod
    def F_parr(
        self,
        k: Quantity,
        mu: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]: ...

    @abstractmethod
    def F_perp(
        self,
        k: Quantity,
        mu: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]: ...

    @abstractmethod
    def detector_noise(self) -> Quantity: ...


class SurveyWindowMixin:
    def Lfield(self: SurveyInterface) -> Quantity:
        zmin, _, zmax = self.get_redshifts()
        Lmin = self.cosmology.comoving(zmin)
        Lmax = self.cosmology.comoving(zmax)
        return Lmax - Lmin

    def Sfield(self: SurveyInterface) -> Quantity:
        _, zmean, _ = self.get_redshifts()
        Omegafield = self.obsparams["Omega_field"].to(u.rad**2).value
        r2 = self.cosmology.comoving(zmean).to(u.Mpc) ** 2
        return r2 * Omegafield

    def Vfield(self: SurveyInterface) -> Quantity:
        Sfield = self.Sfield()
        Lfield = self.Lfield()
        return Sfield * Lfield

    def Wsurvey(
        self: SurveyInterface, q: Quantity, muq: Union[float, np.ndarray]
    ) -> Quantity:
        """Compute the Fourier-transformed sky selection window function"""
        q = np.atleast_1d(q)
        muq = np.atleast_1d(muq)

        qparr = q[:, None, None] * muq[None, :, None]
        qperp = q[:, None, None] * np.sqrt(1 - np.power(muq[None, :, None], 2))

        Sfield = self.Sfield()
        Lperp = np.sqrt(Sfield / np.pi)
        xperp = (qperp * Lperp).to(1).value
        Wperp = Sfield * 2 / xperp * jn(1, xperp)

        Lparr = self.Lfield()
        xparr = (qparr * Lparr).to(1).value
        Wparr = Lparr * spherical_jn(0, xparr / 2)

        Wsurvey = Wperp * Wparr
        return np.squeeze(Wsurvey)


class GalaxySurvey(SurveyWindowMixin, SurveySpecifications):
    def __init__(self, obspars: dict, cosmo: CosmoFunctions) -> None:
        self._obsparams = copy(obspars)
        self._cosmology = cosmo
        self._settings = cosmo.cfg

        self.set_survey_defaults()

    @property
    def cosmology(self) -> CosmoFunctions:
        return self._cosmology

    @cosmology.setter
    def cosmology(self, pcosmology: CosmoFunctions) -> None:
        self._cosmology = pcosmology

    @property
    def obsparams(self) -> dict:
        return self._obsparams

    @obsparams.setter
    def obsparams(self, pobsparams: dict) -> None:
        self._obsparams = pobsparams

    @property
    def settings(self) -> Configuration:
        return self._settings

    @settings.setter
    def settings(self, psettings: Configuration) -> None:
        self._settings = psettings

    def set_survey_defaults(self) -> None:
        self.obsparams.setdefault("Omega_field", 4 * u.deg**2)
        self.obsparams.setdefault("z_mean", np.array([2.833]))
        self.obsparams.setdefault("z_binedges", np.array([2.382, 3.423]))
        self.obsparams.setdefault("spec_err", 0.002)
        self.obsparams.setdefault("ang_res", 0.2 * u.arcsec)
        self.obsparams.setdefault("shot_noise", 0 * u.Mpc**3)

    def get_redshifts(self) -> tuple:
        z = self.obsparams["z_mean"]
        zegeds = self.obsparams["z_binedges"]
        zmin, zmax = zegeds[:-1], zegeds[1:]
        return zmin, z, zmax

    def F_parr(
        self, k: Quantity, mu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)

        _, z, _ = self.get_redshifts()
        sigmapar = (1 + z) / self.cosmology.Hubble(z) * self.obsparams["spec_err"]
        sigmapar = np.atleast_1d(sigmapar)

        logF = (
            -0.5 * (k[:, None, None] * mu[None, :, None] * sigmapar[None, None, :]) ** 2
        )
        return np.squeeze(np.exp(logF.to(1).value))

    def F_perp(
        self, k: Quantity, mu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        _, z, _ = self.get_redshifts()

        sigma_perp = (
            (1 + z)
            * self.cosmology.angdist(z)
            * self.obsparams["ang_res"].to(u.rad).value
        )
        sigma_perp = np.atleast_1d(sigma_perp)

        logF = (
            -0.5
            * (
                k[:, None, None]
                * (1 - mu**2)[None, :, None]
                * sigma_perp[None, None, :]
            )
            ** 2
        )
        return np.squeeze(np.exp(logF.to(1).value))

    def detector_noise(self) -> Quantity:
        return self.obsparams["shot_noise"]


class LIMSuvey(SurveyWindowMixin, SurveySpecifications):
    def __init__(self, obspars: dict, cosmo: CosmoFunctions) -> None:
        self._obsparams = copy(obspars)
        self._cosmology = cosmo
        self._settings = cosmo.settings

        self.set_survey_defaults()

    @property
    def cosmology(self) -> CosmoFunctions:
        return self._cosmology

    @cosmology.setter
    def cosmology(self, pcosmology: CosmoFunctions) -> None:
        self._cosmology = pcosmology

    @property
    def settings(self) -> Configuration:
        return self._settings

    @settings.setter
    def settings(self, psettings: Configuration) -> None:
        self._settings = psettings

    @property
    def obsparams(self) -> dict:
        return self._obsparams

    @obsparams.setter
    def obsparams(self, pobsparams: dict) -> None:
        self._obsparams = pobsparams

    def set_survey_defaults(self) -> None:
        self.obsparams.setdefault("Tsys_NEFD", 40 * u.uK)
        self.obsparams.setdefault("Nfeeds", 19)
        self.obsparams.setdefault("beam_FWHM", 4.1 * u.arcmin)
        self.obsparams.setdefault("nu", 115 * u.GHz)
        self.obsparams.setdefault("dnu", 15 * u.MHz)
        self.obsparams.setdefault("nuObs", 30 * u.GHz)
        self.obsparams.setdefault("Delta_nu", 8 * u.GHz)
        self.obsparams.setdefault("tobs", 1300 * u.h)
        self.obsparams.setdefault("nD", 1)
        self.obsparams.setdefault("Omega_field", 4 * u.deg**2)
        self.obsparams.setdefault("N_FG_par", 1)
        self.obsparams.setdefault("N_FG_perp", 1)
        self.obsparams.setdefault("do_FG_wedge", False)
        self.obsparams.setdefault("a_FG", 0.0)
        self.obsparams.setdefault("b_FG", 0.0)

    def get_redshifts(self) -> tuple:
        # Calculate dz from deltanu
        nu = self.obsparams["nu"]
        nuObs = self.obsparams["nuObs"]
        Delta_nu = self.obsparams["Delta_nu"]
        z = (nu / nuObs - 1).to(1).value
        z_min = (nu / (nuObs + Delta_nu / 2) - 1).to(1).value
        z_max = (nu / (nuObs - Delta_nu / 2) - 1).to(1).value

        return z_min, z, z_max

    def sigma_parr(self) -> Quantity:
        _, z, _ = self.get_redshifts()
        nuObs = self.obsparams["nu"] / (1 + z)

        x = (self.obsparams["dnu"] / nuObs).to(1).value
        y = (1 + z) / self.cosmology.Hubble(z)
        return np.squeeze(x * y)

    def sigma_perp(self) -> Quantity:
        _, z, _ = self.get_redshifts()

        # convert to gaussian variance
        x = self.obsparams["beam_FWHM"].to(u.rad).value / np.sqrt(8 * np.log(2))
        y = self.cosmology.angdist(z) * (1 + z)
        return np.squeeze(x * y)

    def F_parr(
        self, k: Quantity, mu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)

        logF = (
            -0.5
            * np.power(
                k[:, None, None]
                * mu[None, :, None]
                * np.atleast_1d(self.sigma_parr())[None, None, :],
                2,
            )
            .to(1)
            .value
        )

        return np.squeeze(np.exp(logF))

    def F_perp(
        self, k: Quantity, mu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)

        logF = (
            -0.5
            * np.power(
                k[:, None, None]
                * np.sqrt(1 - mu[None, :, None] ** 2)
                * np.atleast_1d(self.sigma_perp())[None, None, :],
                2,
            )
            .to(1)
            .value
        )

        return np.squeeze(np.exp(logF))

    ########################
    # Noise Specifications #
    ########################

    def Npix(self):
        ang_res = self.obsparams["beam_FWHM"]
        Omega_field = self.obsparams["Omega_field"]

        Npix = (Omega_field / ang_res**2).to(1).value
        return np.floor(Npix)

    def tpix(self):
        """Observation time of a single pixel
        observed by one instrument
        """
        Npix = self.Npix()
        tobs = self.obsparams["tobs"]
        return tobs / Npix

    def Vvox(self):
        Npix = self.Npix()
        dnu_FWHM = self.obsparams["dnu"] * np.sqrt(8 * np.log(2))
        Nch = np.floor(self.obsparams["Delta_nu"] / dnu_FWHM).to(1).value
        Nvox = Npix * Nch
        Vvox = self.Vfield() / Nvox
        return Vvox

    def simga_Noise(self) -> Quantity:
        integrated_tobs = (
            self.obsparams["nFeeds"] * self.self.obsparams["nD"] * self.tpix()
        )

        if self.settings.settings["do_Jysr"]:
            sigma_pix = self.obsparams["Tsys_NEFD"] / self.obsparams["beam_FWHM"] ** 2
            return (sigma_pix / np.sqrt(integrated_tobs)).to(u.Jy / u.sr)
        else:
            dnu_FWHM = self.obsparams["dnu"] * np.sqrt(8 * np.log(2))
            sigma_N = (
                self.obsparams["Tsys_NEFD"]
                / np.sqrt(integrated_tobs * dnu_FWHM).to(1).value
            )
            return sigma_N

    def detector_noise(self) -> Quantity:
        return self.simga_Noise() ** 2 * self.Vvox

    def detector_noise_old(self) -> Quantity:
        self.settings.settings["do_Jysr"]
        _, z, _ = self.get_redshifts()
        F1 = (
            self.obsparams["Tsys_NEFD"] ** 2
            * self.obsparams["Omega_field"].to(u.sr).value
            / (self.obsparams["nD"] * self.obsparams["tobs"])
        )
        F2 = self.cosmology.CELERITAS / self.obsparams["nu"]
        F3 = (
            self.cosmology.comoving(z) ** 2
            * (1 + z) ** 2
            / self.cosmology.Hubble(z, physical=True)
        )
        PI = F1 * F2 * F3
        return PI
