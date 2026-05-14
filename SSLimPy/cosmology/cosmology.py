"""
Obtain cosmological functions from the Einstein Boltzmann Code
"""

import types
from copy import deepcopy
from functools import partial
from warnings import warn

import astropy.constants as c
import astropy.units as u
import numpy as np
from scipy.interpolate import RectBivariateSpline as _RectBivariateSpline
from scipy.interpolate import UnivariateSpline as _UnivariateSpline
from scipy.signal import find_peaks
from SSLimPy.interface.config import Configuration
from SSLimPy.utils.utils import *

RectBivariateSpline = partial(_RectBivariateSpline, s=0)
UnivariateSpline = partial(_UnivariateSpline, s=0)


class BoltzmannCode:
    N_EFF = 3.044
    NEUTRINO_MASS_FAC = 94.07

    def __init__(self, cosmopars, cfg: Configuration, code="camb"):
        """
        Constructor method for the class.

        Parameters:
        - cosmopars: The cosmological parameters object to be copied.
        - code: The code to be used (default value is 'camb').
        """
        self.cfg = cfg
        self.settings = cfg.settings

        self.cosmopars = deepcopy(cosmopars)
        self.set_cosmology_defaults()

        if code == "camb":
            import camb as camb

            self.boltzmann_cambpars = cfg.boltzmann_cambpars
            self.camb_results(camb, self.cosmopars)

        elif code == "class":
            from classy import Class

            self.boltzmann_classpars = cfg.boltzmann_classpars
            self.class_results(Class, self.cosmopars)
        else:
            print("other Boltzmann code not implemented yet")
            exit()

    def set_cosmology_defaults(self):
        """
        Fills up default values in the cosmopars dictionary if the values are not found.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        # filling up default values, if value not found in dictionary, then fill it with default value

        # Set default value for Omegam if neither Omegam or omch2 (camb) or omega_cdm (class) are passed
        matter = ["Omegam", "omch2", "omega_cdm", "Omega_cdm"]
        if not any(par in self.cosmopars for par in matter):
            self.cosmopars["Omegam"] = 0.32

        # Set default value for Omegab if neither Omegab or ombh2 or omega_b or Omega_b or 100omega_b are passed
        baryons = ["Omegab", "ombh2", "omega_b", "Omega_b", "100omega_b"]
        if not any(par in self.cosmopars for par in baryons):
            self.cosmopars["Omegab"] = 0.05

        # Set default value for h if neither H0 or h are passed
        localspeed = ["H0", "h"]
        if not any(par in self.cosmopars for par in localspeed):
            self.cosmopars["h"] = 0.67

        # Set default value for ns if it is not found in cosmopars
        primordial_tilt = ["ns", "n_s"]
        if not any(par in self.cosmopars for par in primordial_tilt):
            self.cosmopars["ns"] = 0.96

        primordial_running = ["alpha_s", "nrun"]
        if not any(par in self.cosmopars for par in primordial_running):
            self.cosmopars["alpha_s"] = 0.0

        # Set default value for sigma8 if neither sigma8 or As or logAs or 10^9As are passed
        primordial_amplitude = ["sigma8", "As", "logAs", "10^9As", "ln_A_s_1e10"]
        if not any(par in self.cosmopars for par in primordial_amplitude):
            self.cosmopars["sigma8"] = 0.815583

        # Set default values for w0 and wa if cosmo_model is 'w0waCDM'
        if self.settings["cosmo_model"] == "w0waCDM":
            if not any(par in self.cosmopars for par in ["w", "w0_fld"]):
                self.cosmopars["w0"] = self.cosmopars.get("w0", -1.0)
            if not any(par in self.cosmopars for par in ["wa", "wa_fld"]):
                self.cosmopars["wa"] = self.cosmopars.get("wa", 0.0)

        # Set default value for mnu if Omeganu or omnuh2 or mnu is not found in cosmopars
        if not any(par in self.cosmopars for par in ["Omeganu", "omnuh2", "mnu"]):
            self.cosmopars["mnu"] = self.cosmopars.get(
                "mnu", self.cosmopars.get("m_nu", self.cosmopars.get("M_nu", 0.06))
            )

        # Set default value for Neff if it is not found in cosmopars
        if not any(par in self.cosmopars for par in ["N_ur", "Neff"]):
            self.cosmopars["Neff"] = 3.044

    # Basis Conversion for Class
    def basechange_class(self, cosmopars):
        # transforms cosmopars into cosmopars that can be read by CLASS
        classpars = deepcopy(cosmopars)
        if "H0" in classpars:
            classpars["h"] = classpars["H0"] / 100.0
        h = classpars["h"]

        shareDeltaNeff = self.cfg.settings["share_delta_neff"]
        fidNeff = BoltzmannCode.N_EFF
        Neff = classpars.pop("Neff")
        if shareDeltaNeff:
            # This version does not have the discontinuity at Nur = 1.99
            classpars["N_ur"] = 2.0 / 3.0 * Neff
            g_factor = Neff / 3.0
        else:
            classpars["N_ur"] = Neff - fidNeff / 3.0
            g_factor = fidNeff / 3.0

        neutrino_mass_fac = BoltzmannCode.NEUTRINO_MASS_FAC
        if "mnu" in classpars:
            mnu = classpars.pop("mnu")
            classpars["T_ncdm"] = (4.0 / 11.0) ** (1.0 / 3.0) * g_factor ** (1.0 / 4.0)
            classpars["Omega_ncdm"] = (
                mnu * g_factor ** (0.75) / neutrino_mass_fac / h**2
            )
        elif "Omeganu" in classpars:
            classpars["Omega_ncdm"] = classpars.pop("Omeganu")
        elif "omnuh2" in classpars:
            classpars["Omega_ncdm"] = classpars.pop("omnuh2") / h**2

        if "As" in classpars:
            classpars["A_s"] = classpars.pop("As")
        elif "logAs" in classpars:
            classpars["A_s"] = np.exp(classpars.pop("logAs")) * 1e-10
        elif "10^9As" in classpars:
            classpars["A_s"] = 1e-9 * classpars.pop("10^9As")

        if "ns" in classpars:
            classpars["n_s"] = classpars.pop("ns")

        if "nrun" in classpars:
            classpars["alpha_s"] = classpars.pop("nrun")

        if "w0" in classpars:
            classpars["w0_fld"] = classpars.pop("w0")
            classpars["Omega_Lambda"] = 0

        if "wa" in classpars:
            classpars["wa_fld"] = classpars.pop("wa")

        if "Omegab" in classpars:
            classpars["Omega_b"] = classpars.pop("Omegab")
        elif "ombh2" in classpars:
            classpars["Omega_b"] = classpars.pop("ombh2") / h**2
        elif "100omega_b" in classpars:
            classpars["Omega_b"] = classpars.pop("100omega_b") / 100 / h**2
        elif "omega_b" in classpars:
            classpars["Omega_b"] = classpars.pop("omega_b") / h**2

        if "Omegam" in classpars:
            Om = classpars.pop("Omegam")
            classpars["Omega_cdm"] = Om - classpars["Omega_b"] - classpars["Omega_ncdm"]
        elif "omch2" in classpars:
            classpars["omega_cdm"] = classpars.pop("omch2")

        return classpars

    # Basis Conversion for Camb
    def basechange_camb(self, cosmopars, camb):
        # transforms cosmopars into cosmopars that can be read by CAMB
        cambpars = deepcopy(cosmopars)

        if "h" in cambpars:
            cambpars["H0"] = cambpars.pop("h") * 100
        h = cambpars["H0"] / 100

        shareDeltaNeff = self.cfg.settings["share_delta_neff"]
        cambpars["share_delta_neff"] = shareDeltaNeff
        fidNeff = self.N_EFF
        if "Neff" in cambpars:
            Neff = cambpars.pop("Neff")
            if shareDeltaNeff:
                cambpars["num_nu_massless"] = 2 / 3 * Neff
                g_factor = Neff / 3
            else:
                cambpars["num_nu_massless"] = Neff - fidNeff / 3
                g_factor = fidNeff / 3
        else:
            Neff = cambpars["num_nu_massive"] + cambpars["num_nu_massless"]
        cambpars["standard_neutrino_neff"] = self.N_EFF

        neutrino_mass_fac = self.NEUTRINO_MASS_FAC
        if "mnu" in cambpars:
            cambpars["omnuh2"] = cambpars["mnu"] * g_factor**0.75 / neutrino_mass_fac
        elif "Omeganu" in cambpars:
            cambpars["omnuh2"] = cambpars.pop("Omeganu") * h**2

        if "logAs" in cambpars:
            cambpars["As"] = np.exp(cambpars.pop("logAs")) * 1.0e-10
        elif "10^9As" in cambpars:
            cambpars["As"] = cambpars.pop("10^9As") * 1e-9

        if "n_s" in cambpars:
            cambpars["ns"] = cambpars.pop("n_s")

        if "alpha_s" in cambpars:
            cambpars["nrun"] = cambpars.pop("alpha_s")

        if "w0" in cambpars:
            cambpars["w"] = cambpars.pop("w0")

        if "Omegab" in cambpars:
            cambpars["ombh2"] = cambpars.pop("Omegab") * h**2
        elif "100omega_b" in cambpars:
            cambpars["ombh2"] = cambpars.pop("100omega_b") / 100 * h**2
        elif "omega_b" in cambpars:
            cambpars["ombh2"] = cambpars.pop("omega_b")

        if "Omegam" in cambpars:
            cambpars["omch2"] = (
                cambpars.pop("Omegam") * h**2 - cambpars["ombh2"] - cambpars["omnuh2"]
            )
        elif "Omega_cdm" in cambpars:
            cambpars["omch2"] = cambpars.pop("Omega_cdm") * h**2
        elif "omega_cdm" in cambpars:
            cambpars["omch2"] = cambpars.pop("omega_cdm")

        if "Omegak" in cambpars:
            cambpars["omk"] = cambpars.pop("Omegak")

        rescaleAs = False
        if "sigma8" in cambpars:
            insigma8 = cambpars.pop("sigma8")
            cambpars["As"] = self.settings.get("rescale_ini_As", 2.1e-9)
            rescaleAs = True

        try:
            camb.set_params(**cambpars)
        except camb.CAMBUnknownArgumentError as argument:
            print("Remove parameter from cambparams: " + str(argument))
            raise argument

        if rescaleAs:
            cambpars["As"] = self.rescale_LP(cambpars, camb, insigma8)

        return cambpars

    # CAMB shooting for A_s from sigma8 is to inaccurate so we just call [it] twice
    def rescale_LP(self, cambpars, camb, insigma8):
        cambpars_LP = cambpars.copy()
        ini_As = self.settings["LP_rescale_ini_As"]
        boost = self.settings["LP_rescale_boost"]

        # lower precission of camb calcuation
        cambpars_LP["AccuracyBoost"] = boost
        cambpars_LP["lAccuracyBoost"] = boost
        cambpars_LP["lSampleBoost"] = boost
        cambpars_LP["kmax"] = 20

        # obtain s8 from camb run
        pars = camb.set_params(redshifts=[0.0], **cambpars_LP)
        results = camb.get_results(pars)
        test_sig8 = np.array(results.get_sigma8())

        # rescale linear power spectrum to obtain the s8 asked for
        final_As = ini_As * (insigma8 / test_sig8[-1]) ** 2.0
        return final_As

    def ready_camb(self, cosmopars, camb):

        # Obtain the correct dictionary to be passed to CAMB
        input_cambcosmopars = {
            **self.boltzmann_cambpars["ACCURACY"],
            **self.boltzmann_cambpars["COSMO_SETTINGS"],
        }.copy()
        if self.cfg.settings["nonlinearMatpow"]:
            input_cambcosmopars.update(self.boltzmann_cambpars["NON_LINEAR"])

        input_cambcosmopars.update(cosmopars)
        self.cambcosmopars = self.basechange_camb(input_cambcosmopars, camb)

        cambinstance = camb.set_params(**self.cambcosmopars)

        zmax = self.boltzmann_cambpars["NUMERICS"]["zmax"]
        zsamples = self.boltzmann_cambpars["NUMERICS"]["zsamples"]
        camb_zarray = np.linspace(0.0, zmax, zsamples)[::-1]

        cambinstance.set_matter_power(
            redshifts=camb_zarray,
            k_per_logint=self.cambcosmopars["k_per_logint"],
            kmax=self.cambcosmopars["kmax"],
            accurate_massive_neutrino_transfers=self.cambcosmopars[
                "accurate_massive_neutrino_transfers"
            ],
        )

        ### TEXT VOMIT ###
        if self.cfg.settings["verbosity"] > 1:
            self.recap_camb()
        ##################

        cambres = camb.get_results(cambinstance)
        return cambres, cambinstance

    def ready_class(self, cosmopars, Class):

        input_classcosmopars = {
            **self.boltzmann_classpars["ACCURACY"],
            **self.boltzmann_classpars["COSMO_SETTINGS"],
        }.copy()
        if self.cfg.settings["nonlinearMatpow"]:
            input_classcosmopars.update(self.boltzmann_classpars["NON_LINEAR"])

        input_classcosmopars.update(cosmopars)
        self.classcosmopars = self.basechange_class(input_classcosmopars)

        classres = Class()
        classres.set(self.classcosmopars)

        ### TEXT VOMIT ###
        if self.cfg.settings["verbosity"] > 1:
            self.recap_class()
        ##################

        classres.compute()
        return classres

    def recap_camb(self):
        print("")
        print("----------CAMB Parameters--------")
        print("")
        for key in self.cambcosmopars:
            print("   " + key + ": {}".format(self.cambcosmopars[key]))
        print("")

    def recap_class(self):
        print("")
        print("----------CLASS Parameters--------")
        print("")
        for key in self.classcosmopars:
            print("   " + key + ": {}".format(self.classcosmopars[key]))
        print("")

    def camb_results(self, camb, cosmopars):
        self.results = types.SimpleNamespace()
        cambres, cambinstance = self.ready_camb(cosmopars, camb)

        self.results.h_of_z = UnivariateSpline(
            self.results.zgrid, cambres.h_of_z(self.results.zgrid)
        )
        self.results.ang_dist = UnivariateSpline(
            self.results.zgrid, cambres.angular_diameter_distance(self.results.zgrid)
        )
        self.results.com_dist = UnivariateSpline(
            self.results.zgrid, cambres.comoving_radial_distance(self.results.zgrid)
        )
        self.results.rs_drag = cambres.get_derived_params()["rdrag"]
        self.results.Om_m = UnivariateSpline(
            self.results.zgrid,
            (
                cambres.get_Omega("cdm", z=self.results.zgrid)
                + cambres.get_Omega("baryon", z=self.results.zgrid)
                + cambres.get_Omega("nu", z=self.results.zgrid)
            ),
        )

        self.results.Om_cb = UnivariateSpline(
            self.results.zgrid,
            (
                cambres.get_Omega("cdm", z=self.results.zgrid)
                + cambres.get_Omega("baryon", z=self.results.zgrid)
            ),
        )

        Pk_l, self.results.zgrid, self.results.kgrid = (
            cambres.get_matter_power_interpolator(
                hubble_units=False,
                k_hunit=False,
                var1="delta_tot",
                var2="delta_tot",
                nonlinear=False,
                extrap_kmax=100,
                return_z_k=True,
            )
        )
        self.results.Pk_l = Pk_l.P(self.results.zgrid, self.results.kgrid).T

        Pk_cb_l, _, _ = cambres.get_matter_power_interpolator(
            hubble_units=False,
            k_hunit=False,
            var1="delta_nonu",
            var2="delta_nonu",
            nonlinear=False,
            extrap_kmax=100,
            return_z_k=True,
        )
        self.results.Pk_cb_l = Pk_cb_l.P(self.results.zgrid, self.results.kgrid).T

        if self.cfg.settings["nonlinearMatpow"]:
            Pk_nl, _, _ = cambres.get_matter_power_interpolator(
                hubble_units=False,
                k_hunit=False,
                var1="delta_tot",
                var2="delta_tot",
                nonlinear=True,
                extrap_kmax=100,
                return_z_k=True,
            )
            self.results.Pk_nl = Pk_nl.P(self.results.zgrid, self.results.kgrid).T

            pk_prim = (
                cambinstance.scalar_power(self.results.kgrid)
                * (2.0 * np.pi**2)
                / np.power(self.results.kgrid, 3)
            )
            lgpk_prim = np.log10(pk_prim)
            lgk = np.log10(self.results.kgrid)
            self.results.P_scalar = UnivariateSpline(lgk, lgpk_prim)

            Pk_cross_l = cambres.get_matter_power_interpolator(
                hubble_units=False,
                k_hunit=False,
                var1="delta_nonu",
                var2="delta_nu",
                nonlinear=False,
                extrap_kmax=100,
                return_z_k=False,
            )
            Pk_cross_l = Pk_cross_l.P(self.results.zgrid, self.results.kgrid).T

            Pk_nunu_l = cambres.get_matter_power_interpolator(
                hubble_units=False,
                k_hunit=False,
                var1="delta_nu",
                var2="delta_nu",
                nonlinear=False,
                extrap_kmax=100,
                return_z_k=False,
            )
            Pk_nunu_l = Pk_nunu_l.P(self.results.zgrid, self.results.kgrid).T

            # Calculate the Matter fractions for CB Powerspectrum
            f_cdm = cambres.get_Omega("cdm", z=0) / self.results.Om_m(0)
            f_b = cambres.get_Omega("baryon", z=0) / self.results.Om_m(0)
            f_cb = f_cdm + f_b
            f_nu = 1 - f_cb

            self.results.Pk_cb_nl = (
                self.results.Pk_nl - 2 * Pk_cross_l * f_cb * f_nu - Pk_nunu_l * f_nu**2
            ) / f_cb**2

    def class_results(self, Class, cosmopars):  # Get your CLASS results from here
        self.results = types.SimpleNamespace()
        classres = self.ready_class(cosmopars, Class)
        self.results.h_of_z = np.vectorize(classres.Hubble)
        self.results.ang_dist = np.vectorize(classres.angular_distance)
        self.results.com_dist = np.vectorize(classres.comoving_distance)
        self.results.rs_drag = classres.rs_drag()
        self.results.Om_m = np.vectorize(classres.Om_m)
        self.results.Om_cb = np.vectorize(
            lambda z: classres.Om_cdm(z) + classres.Om_b(z)
        )

        # Calculate the Matter fractions for CB Powerspectrum
        f_cdm = classres.Omega0_cdm() / classres.Omega_m()
        f_b = classres.Omega_b() / classres.Omega_m()
        f_cb = f_cdm + f_b
        f_nu = 1 - f_cb

        ## rows are k, and columns are z
        ## interpolating function Pk_l (k,z)
        Pk_l, self.results.kgrid, zgrid = classres.get_pk_and_k_and_z(
            nonlinear=False,
        )
        self.results.zgrid = zgrid[::-1]
        self.results.Pk_l = Pk_l[:, ::-1]

        Pk_cb_l, _, _ = classres.get_pk_and_k_and_z(
            only_clustering_species=True, nonlinear=False
        )
        self.results.Pk_cb_l = Pk_cb_l[:, ::-1]

        ## interpolating function Pk_nl (k,z)
        if self.cfg.settings["nonlinearMatpow"]:
            Pk_nl, _, _ = classres.get_pk_and_k_and_z(nonlinear=True)
            self.results.Pk_nl = Pk_nl[:, ::-1]

            tk, _, _ = classres.get_transfer_and_k_and_z()
            T_cb = (f_b * tk["d_b"] + f_cdm * tk["d_cdm"]) / f_cb
            T_nu = tk["d_ncdm[0]"]

            pm = classres.get_primordial()
            pk_prim = (
                UnivariateSpline(pm["k [1/Mpc]"], pm["P_scalar(k)"])(self.results.kgrid)
                * (2.0 * np.pi**2)
                / np.power(self.results.kgrid, 3)
            )

            lgpk_prim = np.log10(pk_prim)
            lgk = np.log10(self.results.kgrid)
            self.results.P_scalar = UnivariateSpline(lgk, lgpk_prim)

            Pk_cross_l = T_nu[:, ::-1] * T_cb[:, ::-1] * pk_prim[:, None]
            Pk_nunu_l = T_nu[:, ::-1] * T_nu[:, ::-1] * pk_prim[:, None]

            self.results.Pk_cb_nl = (
                self.results.Pk_nl
                - 2 * Pk_cross_l * f_nu * f_cb
                - Pk_nunu_l * f_nu * f_nu
            ) / f_cb**2


class CosmoFunctions:
    CELERITAS = c.c

    def __init__(
        self,
        cfg: Configuration,
        cosmopars=dict(),
        nuiscance_like=dict(),
        input_type=None,
        cosmology=None,
    ):
        """
        This class is where you can extract all of the EBS results.
        When modifying the code try to keep using the funcitons of this class
        instead of the callables inside of results.
        Will not read cosmopars if cosmology is directly passed
        """
        self.cfg = cfg
        self.settings = cfg.settings

        self.cosmopars = deepcopy(cosmopars)
        self.nuiscance_like = deepcopy(nuiscance_like)

        self.fiducialcosmopars = cfg.fiducialcosmoparams
        if input_type is None:
            input_type = cfg.input_type

        if not cosmology:
            if input_type is None:
                input_type = cfg.input_type
            if input_type == "camb":
                cambresults = BoltzmannCode(cosmopars, cfg, code="camb")
                self.code = "camb"
                self.results = cambresults.results
                self.cosmopars = cambresults.cosmopars
                self.cambcosmopars = cambresults.cambcosmopars
            elif input_type == "class":
                classresults = BoltzmannCode(cosmopars, cfg, code="class")
                self.code = "class"
                self.results = classresults.results
                self.cosmopars = classresults.cosmopars
                self.classcosmopars = classresults.classcosmopars
            else:
                print(input_type, ":  This input type is not implemented yet")
        else:
            self.code = cosmology.code
            self.results = cosmology.results
            self.cosmopars = cosmology.cosmopars
            if self.code == "class":
                self.classcosmopars = cosmology.classcosmopars
            if self.code == "camb":
                self.cambcosmopars = cosmology.cambcosmopars
        self.fullcosmoparams = {**self.cosmopars, **nuiscance_like}

        if self.cfg.settings["k_kind"] == "log":
            k_edge = np.geomspace(
                self.cfg.settings["kmin"],
                self.cfg.settings["kmax"],
                self.cfg.settings["nk"],
            ).to(u.Mpc**-1)
        else:
            k_edge = np.linspace(
                self.cfg.settings["kmin"],
                self.cfg.settings["kmax"],
                self.cfg.settings["nk"],
            ).to(u.Mpc**-1)
        self.k = (k_edge[1:] + k_edge[:-1]) / 2.0

        self.z = np.linspace(
            self.cfg.settings["zmin"],
            self.cfg.settings["zmax"],
            self.cfg.settings["nz"],
        )
        self.create_matter_interp()

    ##############
    # Background #
    ##############

    def Hubble(self, z, physical=False):
        """Hubble function

        Parameters
        ----------
        z     : float
                redshift

        physical: bool
                Default False, if True, return H(z) in (km/s/Mpc).
        Returns
        -------
        float
            Hubble function values (Mpc^-1) at the redshifts of the input redshift

        """
        prefactor = 1
        if physical:
            prefactor = CosmoFunctions.CELERITAS

        hubble = prefactor * self.results.h_of_z(z) * 1 / u.Mpc

        return hubble

    def E_hubble(self, z):
        """E(z) dimensionless Hubble function

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Dimensionless E(z) Hubble function values at the redshifts of the input redshift

        """

        H0 = self.Hubble(0.0)
        Eofz = self.Hubble(z) / H0

        return Eofz

    def h(self) -> float:
        """
        h
        """
        h = (self.Hubble(0, physical=True) / (100 * u.km / u.s / u.Mpc)).to(1)
        return h

    def angdist(self, z):
        """Angular diameter distance

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Angular diameter distance values at the redshifts of the input redshift

        """

        dA = self.results.ang_dist(z) * u.Mpc

        return dA

    def comoving(self, z):
        """Comoving distance

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Comoving distance values at the redshifts of the input redshift

        """

        chi = self.results.com_dist(z) * u.Mpc

        return chi

    def Omegam_of_z(self, z):
        """Omega matter fraction as a function of redshift

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Omega matter (total) at the redshifts of the input redshift `z`


        Note
        -----
        Assumes standard matter evolution
        Implements the following equation:

        .. math::
            Omega_m(z) = Omega_{m,0}*(1+z)^3 / E^2(z)
        """

        omz = self.results.Om_m(z)

        return omz

    def Omega(self, z, tracer="matter"):
        if tracer == "clustering":
            return self.results.Om_cb(z)
        if tracer != "matter":
            warn("Did not recognize tracer: reverted to matter")
        return self.results.Om_m(z)

    def rs_drag(self):
        return self.results.rs_drag * u.Mpc

    #################
    # Power Spectra #
    #################

    def create_matter_interp(self):
        results = self.results
        kgrid = results.kgrid * u.Mpc**-1
        zgrid = results.zgrid

        self.Pk_l = LogLog2DInterpolator(kgrid, zgrid, self.results.Pk_l * u.Mpc**3)
        self.Pk_cb_l = LogLog2DInterpolator(
            kgrid, zgrid, self.results.Pk_cb_l * u.Mpc**3
        )
        if self.cfg.settings["nonlinearMatpow"]:
            self.Pk_nl = LogLog2DInterpolator(
                kgrid, zgrid, self.results.Pk_nl * u.Mpc**3
            )
            self.Pk_cb_nl = LogLog2DInterpolator(
                kgrid, zgrid, self.results.Pk_cb_nl * u.Mpc**3
            )

    def primordial_scalar_pow(self, k):
        lgk = np.log10(k.to(u.Mpc**-1).value)
        pk = np.power(10, self.results.P_scalar(lgk))
        return pk * u.Mpc**3

    def matpow(self, k, z, nonlinear=False, tracer="matter"):
        """Calculates the power spectrum of a given tracer quantity at a specific redshift and wavenumber.

        Parameters
        ----------
        z : array_like
            The redshift of interest.

        k : array_like
            An array of wavenumbers at which to compute the power spectrum. These must be in units of 1/Mpc and
            should be sorted in increasing order.

        nonlinear : bool, optional
            A boolean indicating whether or not to include nonlinear corrections to the matter power spectrum. The default
            value is False.

        tracer : str, optional
            A string indicating which trace quantity to use for computing the power spectrum. If this argument is "matter"
            or anything other than "clustering", the power spectrum functions `Pmm` will be used to compute the power
            spectrum. If the argument is "clustering", the power spectrum function `Pcb` will be used instead. The default
            value is "matter".

        Returns
        -------
        np.ndarray:
            Array containing the calculated power spectrum values.

        Warnings
        --------
        If `tracer` is not "matter" or "clustering", a warning message is printed to the console saying the provided tracer was not
        recognized and the function defaults to using `Pmm` to calculate the power spectrum of matter.
        """
        # Restore Units for Interpolators
        k = np.atleast_1d(k.to(1 / u.Mpc))
        z = np.atleast_1d(z)
        zvec = z[None, :]
        kvec = k[:, None]

        if tracer == "clustering":
            Pk = self.Pcb(kvec, zvec, nonlinear=nonlinear)
        elif tracer == "matter":
            Pk = self.Pmm(kvec, zvec, nonlinear=nonlinear)
        else:
            warn("Did not recognize tracer: reverted to matter")
            Pk = self.Pmm(kvec, zvec, nonlinear=nonlinear)

        ###################################
        # Emulators and Fitting functions #
        ###################################
        if self.cfg.settings["do_pheno_ncdm"]:
            Pk *= self.transfer_ncdm(kvec)

        return np.squeeze(Pk)

    def Transfer(self, k, z, nonlinear=False, tracer="matter"):
        k = np.atleast_1d(k.to(u.Mpc**-1))
        z = np.atleast_1d(z)
        P = np.reshape(
            self.matpow(k, z, nonlinear=nonlinear, tracer=tracer), (*k.shape, *z.shape)
        )
        primordial = self.primordial_scalar_pow(k)[:, None]
        return np.squeeze(np.sqrt(P / primordial))

    def Pmm(self, k, z, nonlinear=False):
        """Compute the power spectrum of the total matter species  (MM) at a given redshift and wavenumber.
        Try to only use Matpow internaly as it is fully vecotrized and handles units correctly

        Args:
            z: The redshift at which to compute the MM power spectrum.
            k: The wavenumber at which to compute the MM power spectrum in 1/Mpc.
            nonlinear (bool, optional): If True, include nonlinear effects in the computation. Default is False.

        Returns:
            float: The value of the MM power spectrum at the given redshift and wavenumber.
        """
        if nonlinear:
            if self.cfg.settings["nonlinearMatpow"]:
                power = self.Pk_nl(k, z)
            else:
                raise AttributeError(
                    "Non-linear power spectrum was not asked for from EBS"
                )
        else:
            power = self.Pk_l(k, z)
        return power

    def Pcb(self, k, z, nonlinear=False):
        """Compute the power spectrum of the clustering matter species  (CB) at a given redshift and wavenumber.
        Try to only use Matpow internaly as it is fully vecotrized and handles units correctly

        Args:
            z: The redshift at which to compute the CB power spectrum.
            k: The wavenumber at which to compute the CB power spectrum in 1/Mpc.
            nonlinear (bool, optional): If True, include nonlinear effects in the computation. Default is False.

        Returns:
            The value of the CB power spectrum at the given redshift and wavenumber.
        """
        if nonlinear:
            if self.cfg.settings["nonlinearMatpow"]:
                power = self.Pk_cb_nl(k, z)
            else:
                raise AttributeError(
                    "Non-linear power spectrum was not asked for from EBS"
                )
        else:
            power = self.Pk_cb_l(k, z)
        return power

    def growth_factor(self, k, z, tracer="matter", nonlinear=False):
        """Compute the scale-depended growth factor D.

        Args:
            k: The wavenumber at which to compute the D.
            z: The redshift at which to compute the D.
            tracer: wheather to compute the D of cb filed or matter
            nonlinear: If True, include nonlinear effects in the computation. Default is False.
        """
        if nonlinear:
            if self.cfg.settings["nonlinearMatpow"]:
                return (
                    self.Pk_cb_nl.D(k, z)
                    if tracer == "clustering"
                    else self.Pk_nl.D(k, z)
                )
            else:
                raise AttributeError(
                    "Non-linear power spectrum was not asked for from EBS"
                )
        return self.Pk_cb_l.D(k, z) if tracer == "clustering" else self.Pk_l.D(k, z)

    def growth_rate(self, k, z, tracer="matter", nonlinear=False):
        """Compute the scale-depended growth rate f.

        Args:
            k: The wavenumber at which to compute the f.
            z: The redshift at which to compute the f.
            tracer: wheather to compute the f of cb filed or matter
            nonlinear: If True, include nonlinear effects in the computation. Default is False.
        """
        if nonlinear:
            if self.cfg.settings["nonlinearMatpow"]:
                return (
                    self.Pk_cb_nl.f(k, z)
                    if tracer == "clustering"
                    else self.Pk_nl.f(k, z)
                )
            else:
                raise AttributeError(
                    "Non-linear power spectrum was not asked for from EBS"
                )
        return self.Pk_cb_l.f(k, z) if tracer == "clustering" else self.Pk_l.f(k, z)

    def P_nw_shape(self, k):
        # Get cosmologyical quantities for the fit
        h = (
            (self.Hubble(0, physical=True) / (100 * u.km * u.s**-1 * u.Mpc**-1))
            .to(1)
            .value
        )
        Om = self.Omega(0, "matter")
        wm = Om * h**2
        wb = self.cosmopars["Omegab"] * h**2

        # This should be changed to the actuall CMB background temp
        theta = 2.7255 / 2.7
        rb = wb / wm
        ns = self.cosmopars["ns"]

        k = k.to(u.Mpc**-1).value
        s = 44.5 * np.log(9.83 / wm) / np.sqrt(1 + 10 * wb ** (3 / 4))
        alpha = 1 - 0.328 * np.log(431 * wm) * rb + 0.38 * np.log(22.3 * wm) * rb**2

        Gamma = (wm / h) * (alpha + (1 - alpha) / (1 + (0.43 * k * s) ** 4))
        q = k / h * theta**2 / Gamma

        L0 = np.log(2 * np.e + 1.8 * q)
        C0 = 14.2 + 731 / (1 + 62.5 * q)
        T_nw = L0 / (L0 + C0 * q**2)
        return T_nw**2 * k**ns

    def nonwiggle_pow(self, k, z, nonlinear=False, tracer="matter"):
        """Calculate the power spectrum at a specific redshift and wavenumber,
        after smoothing to remove baryonic acoustic oscillations (BAO).

        Args:
            z: The redshift of interest.
            k: An array of wavenumbers at which to compute the power
                spectrum.
            nonlinear: Whether to include nonlinear corrections
                to the matter power spectrum. Default is False.
            tracer: Which perturbations to use for computing
                the power spectrum. Options are 'matter' or 'clustering'.
                Default is 'matter'.

        Returns:
            An array of power spectrum values corresponding to the
            input wavenumbers.

        Note:
            This function computes the power spectrum of a given tracer quantity
            at a specific redshift, using the matter power spectrum function `matpow`.
            It then smooths out the BAO signal by constructing two splines trough the inflection points.
            To reduce the dynamic range, the power spectrum is divided
            by the smooth Eisenstein--Hu approximation
        """
        z = np.atleast_1d(z)

        # wave number grids
        kmin_loc = self.settings["smooth_internal_kmin"]
        kmax_loc = self.settings["smooth_internal_kmax"]
        loc_samples = self.settings["smooth_internal_samples"]
        width = self.settings["smooth_internal_width"]
        polyorder = self.settings["smooth_internal_polyorder"]

        kgrid_savgol = np.geomspace(kmin_loc, kmax_loc, loc_samples)
        logkgrid_savgol = np.log(kgrid_savgol.to(u.Mpc**-1).value)

        P = np.reshape(
            self.matpow(kgrid_savgol, z, nonlinear=nonlinear, tracer=tracer),
            (loc_samples, *z.shape),
        )
        uP = P.unit

        P_shape = self.P_nw_shape(kgrid_savgol)
        P_reshape = self.P_nw_shape(k)
        P_shapeless = (P / P_shape[:, None]).value

        Psmoothed = np.empty((*k.shape, *z.shape)) * uP
        for iz, zi in enumerate(z):
            Pprime_inter = UnivariateSpline(
                logkgrid_savgol, P_shapeless[:, iz], s=0, k=polyorder
            ).derivative(1)(logkgrid_savgol)

            logkmin = np.argmin(np.abs(kgrid_savgol - kmin_loc * width))
            logkmax = np.argmin(np.abs(kgrid_savgol - kmax_loc / width))

            logkpeaks, _ = find_peaks(Pprime_inter[:])
            logkvalleys, _ = find_peaks(-Pprime_inter[:])

            ipeaks = [
                *range(logkmin),
                *logkpeaks[(logkpeaks > logkmin) & (logkpeaks < logkmax)],
                *range(logkmax, loc_samples),
            ]
            ivalleys = [
                *range(logkmin),
                *logkvalleys[(logkvalleys > logkmin) & (logkvalleys < logkmax)],
                *range(logkmax, loc_samples),
            ]

            Psl_peaks = UnivariateSpline(
                kgrid_savgol[ipeaks], P_shapeless[ipeaks, iz], s=0, k=polyorder
            )(k)
            Psl_valleys = UnivariateSpline(
                kgrid_savgol[ivalleys], P_shapeless[ivalleys, iz], s=0, k=polyorder
            )(k)
            Psmoothed[:, iz] = 0.5 * (Psl_peaks + Psl_valleys) * P_reshape * uP

        P_locked = np.reshape(
            self.matpow(k, z, nonlinear=nonlinear, tracer=tracer), (*k.shape, *z.shape)
        )
        Psmoothed[np.where((k < kmin_loc) | (k > kmax_loc)), :] = P_locked[
            np.where((k < kmin_loc) | (k > kmax_loc)), :
        ]

        return np.squeeze(Psmoothed)

    def transfer_ncdm(self, ncdmk):
        """
        Transfer function to suppress small-scale power due to non-CDM models as presented in 2404.11609.
        """
        if "f_NL" in self.fullcosmoparams:
            raise ValueError("Cannot have non-zero f_NL and non-CDM.")
        else:
            kcut = self.fullcosmoparams["kcut"]
            slope = self.fullcosmoparams["slope"]
            # make sure k's are in the proper units
            kcut = kcut.to(1.0 / u.Mpc).value
            k = ncdmk

            # Initialize Tk with ones of the same shape as ncdmk
            Tk = np.ones_like(k)
            # Apply the transfer function conditionally
            mask = k > kcut
            Tk[mask] = (k[mask] / kcut) ** (-slope)

        return Tk
