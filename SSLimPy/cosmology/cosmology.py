"""
Obtain cosmological functions from the Einstein Boltzmann Code
"""

import sys
import types
from copy import deepcopy
from warnings import warn

import numpy as np

import astropy.units as u
import astropy.constants as c

from scipy.interpolate import (
    InterpolatedUnivariateSpline,
    RectBivariateSpline,
    UnivariateSpline,
)
from scipy.signal import savgol_filter

# Import SSLimPy functions
sys.path.append("../")
from SSLimPy.interface import config as cfg


class boltzmann_code:
    hardcoded_Neff = 3.043
    hardcoded_neutrino_mass_fac = 94.07

    def __init__(self, cosmopars, code="camb"):
        """
        Constructor method for the class.

        Parameters:
        - cosmopars: The cosmological parameters object to be copied.
        - code: The code to be used (default value is 'camb').
        """

        self.cosmopars = deepcopy(cosmopars)
        self.settings = cfg.settings
        self.set_cosmology_defaults()


        if code == "camb":
            import camb as camb
            self.boltzmann_cambpars = cfg.boltzmann_cambpars
            self.camb_results(camb,self.cosmopars)

        elif code == "class":
            from classy import Class
            self.boltzmann_classpars = cfg.boltzmann_classpars
            self.class_results(Class,self.cosmopars)
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
            self.cosmopars["ns"] = self.cosmopars.get("ns", 0.96)

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
            self.cosmopars["mnu"] = self.cosmopars.get("mnu", self.cosmopars.get("m_nu", self.cosmopars.get("M_nu", 0.06)))

        # Set default value for Neff if it is not found in cosmopars
        if not any(par in self.cosmopars for par in ["N_ur", "Neff"]):
            self.cosmopars["Neff"] = 3.043

        # Set default value for gamma, if it is not found in cosmopars
        # gamma is not used in many places, therefore not needed to add back in cosmopars
        self.gamma = self.cosmopars.get("gamma", 0.545)


    # Basis Conversion for Class
    def basechange_class(self, cosmopars):
        # transforms cosmopars into cosmopars that can be read by CLASS
        shareDeltaNeff = cfg.settings["share_delta_neff"]
        classpars = deepcopy(cosmopars)
        if "h" in classpars:
            classpars["h"] = classpars.pop("h")
            h = classpars["h"]
        if "H0" in classpars:
            classpars["H0"] = classpars.pop("H0")
            h = classpars["H0"] / 100.0

        if "ns" in classpars:
            classpars["n_s"] = classpars.pop("ns")

        if "w0" in classpars:
            classpars["w0_fld"] = classpars.pop("w0")
            classpars["Omega_Lambda"] = 0
        if "wa" in classpars:
            classpars["wa_fld"] = classpars.pop("wa")

        if "As" in classpars:
            classpars["A_s"] = classpars.pop("As")

        Neff = classpars.pop("Neff")
        if shareDeltaNeff:
            classpars['N_ur'] = 2./3.*Neff #This version does not have the discontinuity at Nur = 1.99
            g_factor = Neff/3.
        else:
            classpars['N_ur'] = Neff - boltzmann_code.hardcoded_Neff/3.
            g_factor = boltzmann_code.hardcoded_Neff/3.

        if "mnu" in classpars:
            mnu = classpars.pop("mnu")
            classpars["T_ncdm"] = (4.0 / 11.0) ** (1.0 / 3.0) * g_factor ** (1.0 / 4.0)
            classpars["Omega_ncdm"] = mnu * g_factor ** (0.75) / boltzmann_code.hardcoded_neutrino_mass_fac / h**2
        elif "Omeganu" in classpars:
            classpars["Omega_ncdm"] = classpars.pop("Omeganu")
        elif "omnuh2" in classpars:
            classpars["Omega_ncdm"] = classpars.pop("omnuh2") / h**2

        if "Omegab" in classpars:
            classpars["Omega_b"] = classpars.pop("Omegab")
        if "Omegam" in classpars:
            classpars["Omega_cdm"] = (classpars.pop("Omegam") - classpars["Omega_b"] - classpars["Omega_ncdm"])

        return classpars

    # Basis Conversion for Camb
    def basechange_camb(self, cosmopars, camb):
        shareDeltaNeff = cfg.settings["share_delta_neff"]

        # transforms cosmopars into cosmopars that can be read by CAMB
        cambpars = deepcopy(cosmopars)
        if "h" in cambpars:
            h = cambpars.pop("h")
            h2= h**2
            cambpars["H0"] = h * 100
        if "Omegab" in cambpars:
            cambpars["ombh2"] = cambpars.pop("Omegab") * h2
        if "Omegak" in cambpars:
            cambpars["omk"] = cambpars.pop("Omegak")
        if "w0" in cambpars:
            cambpars["w"] = cambpars.pop("w0")

        if "Neff" in cambpars:
            Neff = cambpars.pop("Neff")
            cambpars["num_nu_massless"] = Neff - boltzmann_code.hardcoded_Neff / 3
        else:
            Neff = cambpars["num_nu_massive"] + cambpars["num_nu_massless"]

        if shareDeltaNeff:
            g_factor = Neff / 3
        else:
            g_factor = boltzmann_code.hardcoded_Neff / 3

        neutrino_mass_fac = 94.07
        h2 = (cambpars["H0"] / 100) ** 2
        if "mnu" in cambpars:
            Onu = cambpars["mnu"] / neutrino_mass_fac * (g_factor) ** 0.75 / h2
            onuh2 = Onu * h2
            cambpars["omnuh2"] = onuh2
        elif "Omeganu" in cambpars:
            cambpars["omnuh2"] = cambpars.pop("Omeganu") * h2
            onuh2 = cambpars["omnuh2"]
        elif "omnuh2" in cambpars:
            onuh2 = cambpars["omnuh2"]
        if "Omegam" in cambpars:  # TO BE GENERALIZED
            cambpars["omch2"] = cambpars.pop("Omegam") * h2 - cambpars["ombh2"] - onuh2

        rescaleAs = False
        if "sigma8" in cambpars:
            insigma8 = cambpars.pop("sigma8")
            cambpars["As"] = 2.1e-9
            rescaleAs = True

        try:
            camb.set_params(**cambpars)
        except camb.CAMBUnknownArgumentError as argument:
            print("Remove parameter from cambparams: ", str(argument))

        if rescaleAs == True:
            cambpars["As"] = self.rescale_LP(cambpars, camb, insigma8)

        return cambpars


    # CAMB shooting for A_s from sigma8 is to inaccurate so we just call [it] twice
    def rescale_LP(self, cambpars, camb, insigma8):
        cambpars_LP = cambpars.copy()
        ini_As = self.settings["LP_rescale_ini_As"]
        boost = self.settings["LP_rescale_boost"]

        #lower precission of camb calcuation
        cambpars_LP["AccuracyBoost"] = boost
        cambpars_LP["lAccuracyBoost"] = boost
        cambpars_LP["lSampleBoost"] = boost
        cambpars_LP["kmax"] = 20

        #obtain s8 from camb run
        pars = camb.set_params(redshifts=[0.0], **cambpars_LP)
        results = camb.get_results(pars)
        test_sig8 = np.array(results.get_sigma8())

        #rescale linear power spectrum to obtain the s8 asked for
        final_As = ini_As * (insigma8 / test_sig8[-1]) ** 2.0
        return final_As

    def ready_camb(self, cosmopars, camb):

        # Obtain the correct dictionary to be passed to CAMB
        input_cambcosmopars = {
            **self.boltzmann_cambpars["ACCURACY"],
            **self.boltzmann_cambpars["COSMO_SETTINGS"],
        }.copy()
        input_cambcosmopars.update(cosmopars)
        self.cambcosmopars = self.basechange_camb(input_cambcosmopars, camb)

        cambinstance = camb.set_params(**self.cambcosmopars)

        self.kmax_pk = self.cambcosmopars["kmax"]
        self.kmin_pk = 1e-4
        zmax = self.boltzmann_cambpars["NUMERICS"]["zmax"]
        zsamples = self.boltzmann_cambpars["NUMERICS"]["zsamples"]
        camb_zarray = np.linspace(0.0, zmax, zsamples)[::-1]

        cambinstance.set_matter_power(
            redshifts=camb_zarray,
            k_per_logint=self.cambcosmopars["k_per_logint"],
            kmax=self.cambcosmopars["kmax"],
            accurate_massive_neutrino_transfers=self.cambcosmopars["accurate_massive_neutrino_transfers"],
        )

        ####TEXT VOMIT###
        self.recap_camb()
        #################

        cambres = camb.get_results(cambinstance)
        return cambres

    def ready_class(self, cosmopars, Class):

        input_classcosmopars = {
            **self.boltzmann_classpars["ACCURACY"],
            **self.boltzmann_classpars["COSMO_SETTINGS"]
        }.copy()
        input_classcosmopars.update(cosmopars)
        self.classcosmopars = self.basechange_class(input_classcosmopars)

        classres = Class()
        classres.set(self.classcosmopars)

        self.kmax_pk = self.classcosmopars["P_k_max_1/Mpc"]
        self.kmin_pk = 1e-4

        ####TEXT VOMIT####
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
        cambres = self.ready_camb(cosmopars,camb)

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
        Pk_nl, zgrid, kgrid = cambres.get_matter_power_interpolator(
            hubble_units=False,
            k_hunit=False,
            var1="delta_tot",
            var2="delta_tot",
            nonlinear=True,
            extrap_kmax=100,
            return_z_k=True,
        )
        Pk_cb_l, zgrid, kgrid = cambres.get_matter_power_interpolator(
            hubble_units=False,
            k_hunit=False,
            var1="delta_nonu",
            var2="delta_nonu",
            nonlinear=False,
            extrap_kmax=100,
            return_z_k=True,
        )

        self.results.Pk_l = RectBivariateSpline(
            self.results.zgrid,
            self.results.kgrid,
            Pk_l.P(self.results.zgrid, self.results.kgrid),
        )
        self.results.Pk_nl = RectBivariateSpline(
            self.results.zgrid,
            self.results.kgrid,
            Pk_nl.P(self.results.zgrid, self.results.kgrid),
        )
        self.results.Pk_cb_l = RectBivariateSpline(
            self.results.zgrid,
            self.results.kgrid,
            Pk_cb_l.P(self.results.zgrid, self.results.kgrid),
        )
        self.results.h_of_z = InterpolatedUnivariateSpline(
            self.results.zgrid, cambres.h_of_z(self.results.zgrid)
        )
        self.results.ang_dist = InterpolatedUnivariateSpline(
            self.results.zgrid, cambres.angular_diameter_distance(self.results.zgrid)
        )
        self.results.com_dist = InterpolatedUnivariateSpline(
            self.results.zgrid, cambres.comoving_radial_distance(self.results.zgrid)
        )
        self.results.Om_m = InterpolatedUnivariateSpline(
            self.results.zgrid,
            (
                cambres.get_Omega("cdm", z=self.results.zgrid)
                + cambres.get_Omega("baryon", z=self.results.zgrid)
                + cambres.get_Omega("nu", z=self.results.zgrid)
            ),
        )

        self.results.Om_cb = InterpolatedUnivariateSpline(
            self.results.zgrid,
            (
                cambres.get_Omega("cdm", z=self.results.zgrid)
                + cambres.get_Omega("baryon", z=self.results.zgrid)
            ),
        )

        # Calculate the Non linear cb power spectrum using Gabrieles Approximation
        f_cdm = cambres.get_Omega("cdm", z=0) / self.results.Om_m(0)
        f_b = cambres.get_Omega("baryon", z=0) / self.results.Om_m(0)
        f_cb = f_cdm + f_b
        f_nu = 1 - f_cb
        Pk_cross_l = cambres.get_matter_power_interpolator(
            hubble_units=False,
            k_hunit=False,
            var1="delta_nonu",
            var2="delta_nu",
            nonlinear=False,
            extrap_kmax=100,
            return_z_k=False,
        )
        Pk_nunu_l = cambres.get_matter_power_interpolator(
            hubble_units=False,
            k_hunit=False,
            var1="delta_nu",
            var2="delta_nu",
            nonlinear=False,
            extrap_kmax=100,
            return_z_k=False,
        )
        Pk_cb_nl = (
            1
            / f_cb**2
            * (
                Pk_nl.P(self.results.zgrid, self.results.kgrid)
                - 2 * Pk_cross_l.P(self.results.zgrid, self.results.kgrid) * f_cb * f_nu
                - Pk_nunu_l.P(self.results.zgrid, self.results.kgrid) * f_nu**2
            )
        )
        self.results.Pk_cb_nl = RectBivariateSpline(
            self.results.zgrid, self.results.kgrid, Pk_cb_nl
        )

        P_kz_0 = self.results.Pk_l(0.0, self.results.kgrid)
        D_g_norm_kz = np.sqrt(
            self.results.Pk_l(self.results.zgrid, self.results.kgrid) / P_kz_0
        )

        self.results.D_growth_zk = RectBivariateSpline(
            self.results.zgrid, self.results.kgrid, (D_g_norm_kz), kx=3, ky=3
        )

        P_cb_kz_0 = self.results.Pk_cb_l(0.0, self.results.kgrid)
        D_g_cb_norm_kz = np.sqrt(
            self.results.Pk_cb_l(self.results.zgrid, self.results.kgrid) / P_cb_kz_0
        )
        self.results.D_growth_cb_zk = RectBivariateSpline(
            self.results.zgrid, self.results.kgrid, (D_g_cb_norm_kz), kx=3, ky=3
        )

        def f_deriv(k_array, k_fix=False, fixed_k=1e-3):
            z_array = self.results.zgrid
            if k_fix:
                k_array = np.full((len(k_array)), fixed_k)
            ## Generates interpolaters D(z) for varying k values
            D_z = np.array(
                [
                    UnivariateSpline(
                        z_array, self.results.D_growth_zk(z_array, kk), s=0
                    )
                    for kk in k_array
                ]
            )

            ## Generates arrays f(z) for varying k values
            f_z = np.array(
                [
                    -(1 + z_array) / D_zk(z_array) * (D_zk.derivative())(z_array)
                    for D_zk in D_z
                ]
            )
            return f_z, z_array

        f_z_k_array, z_array = f_deriv(self.results.kgrid)
        self.results.f_growthrate_zk = RectBivariateSpline(
            z_array, self.results.kgrid, f_z_k_array.T
        )

        def f_cb_deriv(k_array, k_fix=False, fixed_k=1e-3):
            z_array = self.results.zgrid
            if k_fix:
                k_array = np.full((len(k_array)), fixed_k)
            ## Generates interpolaters D(z) for varying k values
            D_cb_z = np.array(
                [
                    UnivariateSpline(
                        z_array, self.results.D_growth_cb_zk(z_array, kk), s=0
                    )
                    for kk in k_array
                ]
            )

            ## Generates arrays f(z) for varying k values
            f_cb_z = np.array(
                [
                    -(1 + z_array) / D_cb_zk(z_array) * (D_cb_zk.derivative())(z_array)
                    for D_cb_zk in D_cb_z
                ]
            )
            return f_cb_z, z_array

        f_cb_z_k_array, z_array = f_cb_deriv(self.results.kgrid)
        self.results.f_growthrate_cb_zk = RectBivariateSpline(
            z_array, self.results.kgrid, f_cb_z_k_array.T
        )

        def get_sigma8(z_range):
            R = 8.0 / (cambres.Params.H0 / 100.0)
            k = np.linspace(self.kmin_pk, self.kmax_pk, 10000)
            sigma_z = np.empty_like(z_range)
            pkz = self.results.Pk_l(z_range, k)
            for i in range(len(sigma_z)):
                integrand = (
                    9
                    * (k * R * np.cos(k * R) - np.sin(k * R)) ** 2
                    * pkz[i]
                    / k**4
                    / R**6
                    / 2
                    / np.pi**2
                )
                sigma_z[i] = np.sqrt(np.trapz(integrand, k))
            sigm8_z_interp = UnivariateSpline(z_range, sigma_z, s=0)
            return sigm8_z_interp

        def get_sigma8_cb(z_range):
            R = 8.0 / (cambres.Params.H0 / 100.0)
            k = np.linspace(self.kmin_pk, self.kmax_pk, 10000)
            sigma_cb_z = np.empty_like(z_range)
            pk_cb_z = self.results.Pk_cb_l(z_range, k)
            for i in range(len(sigma_cb_z)):
                integrand = (
                    9
                    * (k * R * np.cos(k * R) - np.sin(k * R)) ** 2
                    * pk_cb_z[i]
                    / k**4
                    / R**6
                    / 2
                    / np.pi**2
                )
                sigma_cb_z[i] = np.sqrt(np.trapz(integrand, k))
            sigm8_cb_z_interp = UnivariateSpline(z_range, sigma_cb_z, s=0)
            return sigm8_cb_z_interp

        self.results.s8_cb_of_z = get_sigma8_cb(self.results.zgrid)
        self.results.s8_of_z = get_sigma8(self.results.zgrid)

        if self.cambcosmopars["Want_CMB"]:
            powers = cambres.get_cmb_power_spectra(CMB_unit="muK")
            self.results.camb_cmb = powers["total"]

    def class_results(self, Class, cosmopars):  # Get your CLASS results from here
        self.results = types.SimpleNamespace()
        classres = self.ready_class(cosmopars, Class)
        self.results.h_of_z = np.vectorize(classres.Hubble)
        self.results.ang_dist = np.vectorize(classres.angular_distance)
        self.results.com_dist = np.vectorize(classres.comoving_distance)
        h = classres.h()
        self.results.s8_of_z = np.vectorize(lambda zz: classres.sigma(R=8 / h, z=zz))
        self.results.s8_cb_of_z = np.vectorize(
            lambda zz: classres.sigma_cb(R=8 / h, z=zz)
        )
        self.results.Om_m = np.vectorize(classres.Om_m)
        self.results.Om_cb= np.vectorize(lambda z: classres.Om_cdm(z)+classres.Om_b(z))

        # Calculate the Matter fractions for CB Powerspectrum
        f_cdm = classres.Omega0_cdm() / classres.Omega_m()
        f_b = classres.Omega_b() / classres.Omega_m()
        f_cb = f_cdm + f_b
        f_nu = 1 - f_cb

        ## rows are k, and columns are z
        ## interpolating function Pk_l (k,z)
        Pk_l, k, z = classres.get_pk_and_k_and_z(nonlinear=False)
        Pk_cb_l, k, z = classres.get_pk_and_k_and_z(
            only_clustering_species=True, nonlinear=False
        )
        self.results.Pk_l = RectBivariateSpline(
            z[::-1], k, (np.flip(Pk_l, axis=1)).transpose()
        )
        # self.results.Pk_l = lambda z,k: [np.array([classres.pk_lin(kval,z) for kval in k])]
        self.results.Pk_cb_l = RectBivariateSpline(
            z[::-1], k, (np.flip(Pk_cb_l, axis=1)).transpose()
        )
        # self.results.Pk_cb_l = lambda z,k: [np.array([classres.pk_cb_lin(kval,z) for kval in k])]

        self.results.kgrid = k
        self.results.zgrid = z[::-1]

        ## interpolating function Pk_nl (k,z)
        Pk_nl, k, z = classres.get_pk_and_k_and_z(nonlinear=cfg.settings["nonlinear"])
        self.results.Pk_nl = RectBivariateSpline(
            z[::-1], k, (np.flip(Pk_nl, axis=1)).transpose()
        )

        tk, k, z = classres.get_transfer_and_k_and_z()
        T_cb = (f_b * tk["d_b"] + f_cdm * tk["d_cdm"]) / f_cb
        T_nu = tk["d_ncdm[0]"]

        pm = classres.get_primordial()
        pk_prim = (
            UnivariateSpline(pm["k [1/Mpc]"], pm["P_scalar(k)"])(k)
            * (2.0 * np.pi**2)
            / np.power(k, 3)
        )

        pk_cnu = T_nu * T_cb * pk_prim[:, None]
        pk_nunu = T_nu * T_nu * pk_prim[:, None]
        Pk_cb_nl = (
            1.0 / f_cb**2 * (Pk_nl - 2 * pk_cnu * f_nu * f_cb - pk_nunu * f_nu * f_nu)
        )

        self.results.Pk_cb_nl = RectBivariateSpline(
            z[::-1], k, (np.flip(Pk_cb_nl, axis=1)).transpose()
        )

        def create_growth():
            z_ = self.results.zgrid
            pk_flipped = np.flip(Pk_l, axis=1).T
            D_growth_zk = RectBivariateSpline(
                z_, k, np.sqrt(pk_flipped / pk_flipped[0, :])
            )
            return D_growth_zk

        self.results.D_growth_zk = create_growth()

        def f_deriv(k_array, k_fix=False, fixed_k=1e-3):
            z_array = np.linspace(0, classres.pars["z_max_pk"], 100)
            if k_fix:
                k_array = np.full((len(k_array)), fixed_k)
            ## Generates interpolaters D(z) for varying k values
            D_z = np.array(
                [
                    UnivariateSpline(
                        z_array, self.results.D_growth_zk(z_array, kk), s=0
                    )
                    for kk in k_array
                ]
            )

            ## Generates arrays f(z) for varying k values
            f_z = np.array(
                [
                    -(1 + z_array) / D_zk(z_array) * (D_zk.derivative())(z_array)
                    for D_zk in D_z
                ]
            )
            return f_z, z_array

        f_z_k_array, z_array = f_deriv(self.results.kgrid)
        f_g_kz = RectBivariateSpline(z_array, self.results.kgrid, f_z_k_array.T)
        self.results.f_growthrate_zk = f_g_kz

        def create_growth_cb():
            z_ = self.results.zgrid
            pk_flipped = np.flip(Pk_cb_l, axis=1).T
            D_growth_zk = RectBivariateSpline(
                z_, k, np.sqrt(pk_flipped / pk_flipped[0, :])
            )
            return D_growth_zk

        self.results.D_growth_cb_zk = create_growth_cb()

        def f_cb_deriv(k_array, k_fix=False, fixed_k=1e-3):
            z_array = np.linspace(0, classres.pars["z_max_pk"], 100)
            if k_fix:
                k_array = np.full((len(k_array)), fixed_k)
            ## Generates interpolaters D(z) for varying k values
            D_cb_z = np.array(
                [
                    UnivariateSpline(
                        z_array, self.results.D_growth_cb_zk(z_array, kk), s=0
                    )
                    for kk in k_array
                ]
            )

            ## Generates arrays f(z) for varying k values
            f_cb_z = np.array(
                [
                    -(1 + z_array) / D_cb_zk(z_array) * (D_cb_zk.derivative())(z_array)
                    for D_cb_zk in D_cb_z
                ]
            )
            return f_cb_z, z_array

        f_cb_z_k_array, z_array = f_cb_deriv(self.results.kgrid)
        f_g_cb_kz = RectBivariateSpline(z_array, self.results.kgrid, f_cb_z_k_array.T)
        self.results.f_growthrate_cb_zk = f_g_cb_kz


class cosmo_functions:
    celeritas = c.c

    def __init__(self, cosmopars, input=None):
        self.settings = cfg.settings
        self.fiducialcosmopars = cfg.fiducialcosmoparams
        self.input = input
        if input is None:
            input = cfg.input_type
        if input == "camb":
            cambresults = boltzmann_code(cosmopars, code="camb")
            self.code = "camb"
            self.results = cambresults.results
            self.kgrid = cambresults.results.kgrid
            self.cosmopars = cambresults.cosmopars
            self.cambcosmopars = cambresults.cambcosmopars
        elif input == "class":
            classresults = boltzmann_code(cosmopars, code="class")
            self.code = "class"
            self.results = classresults.results
            self.kgrid = classresults.results.kgrid
            self.cosmopars = classresults.cosmopars
            self.classcosmopars = classresults.classcosmopars
        else:
            print(input, ":  This input type is not implemented yet")

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
            prefactor = cosmo_functions.celeritas

        hubble = prefactor * self.results.h_of_z(z) * 1/u.Mpc

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

    def matpow(self, z, k, nonlinear=False, tracer="matter"):
        """Calculates the power spectrum of a given tracer quantity at a specific redshift and wavenumber.

        Parameters
        ----------
        z : float
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
        if tracer == "clustering":
            Ps = self.Pcb(z, k, nonlinear=nonlinear) * u.Mpc**3
        elif tracer != "matter":
            warn("Did not recognize tracer: reverted to matter")

        Ps  = self.Pmm(z, k, nonlinear=nonlinear) * u.Mpc**3
        return Ps

    def Pmm(self, z, k, nonlinear=False):
        """Compute the power spectrum of the total matter species  (MM) at a given redshift and wavenumber.

        Args:
            z: The redshift at which to compute the MM power spectrum.
            k: The wavenumber at which to compute the MM power spectrum in 1/Mpc.
            nonlinear (bool, optional): If True, include nonlinear effects in the computation. Default is False.

        Returns:
            float: The value of the MM power spectrum at the given redshift and wavenumber.
        """
        if nonlinear is True:
            power = self.results.Pk_nl(z, k, grid=False)
        elif nonlinear is False:
            power = self.results.Pk_l(z, k, grid=False)
        return power

    def Pcb(self, z, k, nonlinear=False):
        """Compute the power spectrum of the clustering matter species  (CB) at a given redshift and wavenumber.

        Args:
            z: The redshift at which to compute the CB power spectrum.
            k: The wavenumber at which to compute the CB power spectrum in 1/Mpc.
            nonlinear (bool, optional): If True, include nonlinear effects in the computation. Default is False.

        Returns:
            The value of the CB power spectrum at the given redshift and wavenumber.
        """
        if nonlinear is True:
            power = self.results.Pk_cb_nl(z, k, grid=False)
        elif nonlinear is False:
            power = self.results.Pk_cb_l(z, k, grid=False)
        return power

    def nonwiggle_pow(self, z, k, nonlinear=False, tracer="matter"):
        """Calculate the power spectrum at a specific redshift and wavenumber,
        after smoothing to remove baryonic acoustic oscillations (BAO).

        Args:
            z: The redshift of interest.
            k: An array of wavenumbers at which to compute the power
                spectrum. Must be in units of Mpc^-1/h. Should be sorted in
                increasing order.
            nonlinear: Whether to include nonlinear corrections
                to the matter power spectrum. Default is False.
            tracer: Which perturbations to use for computing
                the power spectrum. Options are 'matter' or 'clustering'.
                Default is 'matter'.

        Returns:
            An array of power spectrum values corresponding to the
            input wavenumbers. Units are (Mpc/h)^3.

        Note:
            This function computes the power spectrum of a given tracer quantity
            at a specific redshift, using the matter power spectrum function
            `matpow`. It then applies a Savitzky-Golay filter to smooth out the
            BAO features in the power spectrum. This is done by first taking the
            natural logarithm of the power spectrum values at a set of logarithmic
            wavenumbers spanning from `kmin_loc` to `kmax_loc`. The smoothed power
            spectrum is then returned on a linear (not logarithmic) grid of
            wavenumbers given by the input array `k`.
        """
        unitsf = self.cosmopars["h"]
        kmin_loc = unitsf * self.settings["savgol_internalkmin"]
        kmax_loc = unitsf * np.max(self.kgrid)
        loc_samples = self.settings["savgol_internalsamples"]
        log_kgrid_loc = np.linspace(np.log(kmin_loc), np.log(kmax_loc), loc_samples)
        poly_order = self.settings["savgol_polyorder"]
        dlnk_loc = np.mean(log_kgrid_loc[1:] - log_kgrid_loc[0:-1])
        savgol_width = self.settings["savgol_width"]
        n_savgol = int(np.round(savgol_width / np.log(1 + dlnk_loc)))

        P = self.matpow(z, np.exp(log_kgrid_loc), nonlinear=nonlinear, tracer=tracer).flatten()
        uP= P.unit

        intp_p = InterpolatedUnivariateSpline(
            log_kgrid_loc,
            np.log(P.value),
            k=1,
        )
        pow_sg = savgol_filter(intp_p(log_kgrid_loc), n_savgol, poly_order)
        intp_pnw = InterpolatedUnivariateSpline(
            np.exp(log_kgrid_loc), np.exp(pow_sg), k=1
        )
        return intp_pnw(k) * uP

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

    def sigma8_of_z(self, z, tracer="matter"):
        """sigma_8

        Parameters
        ----------
        z     : float
                redshift
        tracer: String
                either 'matter' if you want sigma_8 calculated from the total matter power spectrum or 'clustering' if you want it from the Powerspectrum with massive neutrinos substracted
        Returns
        -------
        float
            The Variance of the matter perturbation smoothed over a scale of 8 Mpc/h

        """
        if tracer == "clustering":
            return self.results.s8_cb_of_z(z)
        if tracer != "matter":
            warn("Did not recognize tracer: reverted to matter")
        return self.results.s8_of_z(z)

    def sigmaR_of_z(self, z, R,tracer="matter"):
        """sigma_8

        Parameters
        ----------
        z     : float
                redshift
        R     : float, numpy.ndarray
                Radii
        tracer: String
                either 'matter' if you want sigma_8 calculated from the total matter power spectrum or 'clustering' if you want it from the Powerspectrum with massive neutrinos substracted
        Returns
        -------
        float
            The Variance of the matter perturbation smoothed over a scale of R in Mpc

        """
        R = np.atleast_1d(R)[None,:]
        k = self.results.kgrid / u.Mpc


        Pk = self.matpow(z,k,tracer=tracer)[:,None]
        
        x = (k[:,None] * R).to(1)

        #Get Sigma window function
        W = 3 /np.power(x,3)*(np.sin(x*u.rad)-x*np.cos(x*u.rad))
        for ix, xi in enumerate(x):
            if xi<0.01:
                W[ix]=1-xi**2/10

        Integr= np.power(k[:,None]*W,2)*Pk/(2*np.pi**2)
        return np.sqrt(np.trapz(Integr,k,axis=0))

    def growth(self, z, k=None):
        """Growth factor

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Growth factor values at the redshifts of the input redshift

        """
        if k is None:
            k = 0.0001
        Dg = self.results.D_growth_zk(z, k, grid=False)

        return Dg

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
        omz = 0
        if self.input == "external":
            omz = (self.cosmopars["Omegam"] * (1 + z) ** 3) / self.E_hubble(z) ** 2
        else:
            omz = self.results.Om_m(z)

        return omz

    def Omega(self, z, tracer = "matter"):
        if tracer == "clustering":
            return self.results.Om_cb(z)
        if tracer != "matter":
            warn("Did not recognize tracer: reverted to matter")
        return self.results.Om_m(z)

    def f_growthrate(self, z, k=None, gamma=False, tracer="matter"):
        """Growth rate in LCDM gamma approximation

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Growth rate values at the redshifts of the input redshift,
            using self.gamma as gamma value.


        Note
        -----
        Implements the following equation:

        .. math::
            f(z) = Omega_m(z)^{\gamma}
        """
        if k is None:
            k = 0.0001

        if tracer == "clustering":
            fg = self.results.f_growthrate_cb_zk(z, k, grid=False)
            return fg
        if tracer != "matter":
            warn("Did not recognize tracer: reverted to matter")

        if gamma is False:
            fg = self.results.f_growthrate_zk(z, k, grid=False)
        else:
            # Assumes standard Omega_matter evolution in z
            fg = np.power(self.Omegam_of_z(z), self.gamma)

        return fg

    def fsigma8_of_z(self, z, k=None, gamma=False, tracer="matter"):
        """Growth rate in LCDM gamma approximation

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Growth rate values at the redshifts of the input redshift,
            using self.gamma as gamma value.


        Note
        -----
        Implements the following equation:

        .. math::
            f(z) = Omega_m(z)^{\gamma}
        """
        # Assumes standard Omega_matter evolution in z
        fs8 = self.f_growthrate(z, k, gamma, tracer=tracer) * self.sigma8_of_z(
            z, tracer=tracer
        )

        return fs8

    def SigmaMG(self, z, k):
        Sigma = np.array(1)
        if self.settings["activateMG"] == "late-time":
            E11 = self.cosmopars["E11"]
            E22 = self.cosmopars["E22"]
            # TODO: Fix for non-flat models
            Omega_DE = 1 - self.Omegam_of_z(z)
            mu = 1 + E11 * Omega_DE
            eta = 1 + E22 * Omega_DE
            Sigma = (mu / 2) * (1 + eta)
        elif (
            self.settings["external_activateMG"] is True
            or self.settings["activateMG"] == "external"
        ):
            Sigma = self.results.SigWL_zk(z, k, grid=False)

        return Sigma

    def cmb_power(self, lmin, lmax, obs1, obs2):
        if self.code == "camb":
            if self.cambcosmopars.Want_CMB:
                print("CMB Spectrum not computed")
                return
        elif self.code == "class":
            if "tCl" in self.classcosmopars["output"]:
                print("CMB Spectrum not computed")
                return
        else:
            ells = np.arange(lmin, lmax)

            norm_fac = 2 * np.pi / (ells * (ells + 1))

            if obs1 + obs2 == "CMB_TCMB_T":
                cls = norm_fac * self.results.camb_cmb[lmin:lmax, 0]
            elif obs1 + obs2 == "CMB_ECMB_E":
                cls = norm_fac * self.results.camb_cmb[lmin:lmax, 1]
            elif obs1 + obs2 == "CMB_BCMB_B":
                cls = norm_fac * self.results.camb_cmb[lmin:lmax, 2]
            elif (obs1 + obs2 == "CMB_TCMB_E") or (obs1 + obs2 == "CMB_ECMB_T"):
                cls = norm_fac * self.results.camb_cmb[lmin:lmax, 3]
            else:
                cls = np.array([0.0] * len(ells))

            return cls
