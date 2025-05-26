import numpy as np
import astropy.units as u

from numba import njit, prange
from copy import deepcopy
from scipy.special import sici

from SSLimPy.cosmology.cosmology import CosmoFunctions
from SSLimPy.cosmology.fitting_functions import coevolution_bias as cb
from SSLimPy.cosmology.fitting_functions import halo_mass_functions as HMF
from SSLimPy.interface import config as cfg
from SSLimPy.utils.utils import *


class HaloModel:
    """Computation of Halo model ingredients used to compute the nonlinear power spectrum"""

    def __init__(
        self,
        cosmo: CosmoFunctions,
        halopars: dict = dict(),
    ):
        self.cosmology = cosmo
        self.haloparams = deepcopy(halopars)
        self._set_halo_defaults()

        # Units
        self.hubble = self.cosmology.h()
        self.Mpch = u.Mpc / self.hubble
        self.Msunh = u.Msun / self.hubble

        # Densities and collapse tracer
        self.tracer = self.haloparams["halo_tracer"]
        self.rho_crit = 2.77536627e11 * (self.Msunh * self.Mpch**-3).to(
            u.Msun * u.Mpc**-3
        )
        self.rho_tracer = self.rho_crit * self.cosmology.Omega(0, tracer=self.tracer)

        # Internal grids
        self.R = np.geomspace(
            self.haloparams["Rmin"],
            self.haloparams["Rmax"],
            self.haloparams["nR"],
        ).to(u.Mpc)
        self.M = (4 * np.pi / 3 * self.rho_tracer * self.R**3).to(u.Msun)
        self.Mmin = min(self.M)
        self.Mmax = max(self.M)

        if cfg.settings["k_kind"] == "log":
            k_edge = np.geomspace(
                cfg.settings["kmin"],
                cfg.settings["kmax"],
                cfg.settings["nk"],
            ).to(u.Mpc**-1)
        else:
            k_edge = np.linspace(
                cfg.settings["kmin"],
                cfg.settings["nk"],
            ).to(u.Mpc**-1)
        self.k = (k_edge[1:] + k_edge[:-1]) / 2.0

        self.z = np.linspace(
            cfg.settings["zmin"],
            cfg.settings["zmax"],
            cfg.settings["nz"],
        )

        self.sigmaR_lut, self.dsigmaR_lut = self._create_sigmaR_lookuptable()
        self._init_halo_mass_function()
        self._init_bias_function()

        # !Without corrections for non-Gaussianity!
        # bias function
        self.delta_crit = 1.686
        self._b_of_M = getattr(self._bias_function, self.haloparams["bias_model"])
        # use halobias to compute the bias with all corrections

        # halo mass function
        self._dn_dM_of_M = getattr(
            self._halo_mass_function, self.haloparams["hmf_model"]
        )
        # use halomassfunction to compute the bias with all corrections
        G, n, logc = self._create_concentration_lookuptable()
        self.conc_G_lut = G
        self.conc_n_lut = n
        self.conc_logc_lut = logc

    def _set_halo_defaults(self):
        """
        Fills up default values in the halo dictionary if the values are not found.
        """
        self.haloparams.setdefault("halo_tracer", "clustering")
        self.haloparams.setdefault("hmf_model", "ST")
        self.haloparams.setdefault("bias_model", "ST99")
        self.haloparams.setdefault("bias_par", {})
        self.haloparams.setdefault("v_of_M", None)
        self.haloparams.setdefault("Rmin", 1.0e-4 * u.Mpc)
        self.haloparams.setdefault("Rmax", 1.5e3 * u.Mpc)
        self.haloparams.setdefault("nR", 64)
        self.haloparams.setdefault("alpha_iSigma", 3)
        self.haloparams.setdefault("tol_sigma", 1e-4)
        self.haloparams.setdefault("concentration", "Diemer19")
        self.haloparams.setdefault("bloating", False)
        self.haloparams.setdefault("transition_smoothing", False)

    def _init_halo_mass_function(self):
        """
        Initialise computation of halo mass function.
        Raises error if model given by hmf_model does not exist.
        """
        hmf_model = self.haloparams["hmf_model"]
        self._halo_mass_function = HMF.halo_mass_functions(self)

        if not hasattr(self._halo_mass_function, hmf_model):
            raise ValueError(hmf_model + " not found in halo_mass_functions.py")

    def _init_bias_function(self):
        """
        Initialise computation of bias function.
        Raises error if  model given by bias_model does not exist.
        """
        bias_name = self.haloparams["bias_model"]
        self._bias_function = cb.coevolution_bias(self)
        if not hasattr(self._bias_function, bias_name):
            raise ValueError(bias_name + " not found in bias_fitting_functions.py")

    def _create_sigmaR_lookuptable(self):
        """
        Create the sigmaR, dsigmaR look up table used to obtain quantities on interpolated grid
        """
        # Unitless linear power spectrum
        R = self.R
        z = self.z
        kinter = self.k.to(u.Mpc**-1)
        Dkinter = (
            (
                4
                * np.pi
                * (kinter[:, None] / (2 * np.pi)) ** 3
                * self.cosmology.matpow(kinter, z, nonlinear=False, tracer=self.tracer)
            )
            .to(1)
            .value
        )

        # obtain sigma grids
        sigmaR, dsigmaR = sigmas_of_R_and_z(
            R.value,
            z,
            kinter.value,
            Dkinter,
            self.haloparams["alpha_iSigma"],
            self.haloparams["tol_sigma"],
        )
        return sigmaR, dsigmaR

    def _create_concentration_lookuptable(self):
        """
        concentration-mass relation for the NFW profile.
        Following Diemer & Joyce (2019)
        c = R_delta / r_s (the scale radius, not the sound horizon)
        """
        # Numerical parameters used for bisection
        n_G = 80
        n_n = 40
        n_c = 80

        n = np.linspace(-4.0, 0.0, n_n)
        c = np.geomspace(0.1, 1e3, n_c)

        mu = np.log(1 + c) - c / (1.0 + c)
        lhs = np.log10(c[:, None] / mu[:, None]**((5.0 + n) / 6.0))

        # At very low concentration and shallow slopes, the LHS begins to rise again. This will cause
        # issues with the inversion. We set those parts of the curve to the minimum concentration of
        # a given n bin.
        mask_ascending = np.ones_like(lhs, bool)
        mask_ascending[:-1, :] = (np.diff(lhs, axis = 0) > 0.0)

        # Create a table of c as a function of G and n. First, use the absolute min and max of G as
        # the table range
        G_min = np.min(lhs)
        G_max = np.max(lhs)
        G = np.linspace(G_min, G_max, n_G)

        logc_table = np.empty((n_G, n_n), float)
        mins = np.zeros_like(n)
        maxs = np.zeros_like(n)
        for i in range(n_n):

            # We interpolate only the ascending values to get c(G)
            mask_ = mask_ascending[:, i]
            lhs_ = lhs[mask_, i]
            mins[i] = np.min(lhs_)
            maxs[i] = np.max(lhs_)

            # Not all G exist for all n
            mask = (G >= mins[i]) & (G <= maxs[i])
            interp = linear_interpolate(lhs_, np.log10(c[mask_]), G[mask])
            logc_table[mask, i] = interp

            # Do constant extrapolation
            mask_low = (G < mins[i])
            logc_table[mask_low, i] = np.min(interp)
            mask_high = (G > maxs[i])
            logc_table[mask_high, i] = np.max(interp)

        return G, n, logc_table

    def _compute_sigma(self, R, z, ingrnd, tracer="matter", moment=0):
        """Computes sigma if specifically asked for value not present in LUT"""
        R = np.atleast_1d(R.to(u.Mpc).value)
        z = np.atleast_1d(z).astype(float)
        # Save shapes
        Rs = R.shape
        zs = z.shape
        R = R.flatten()
        z = z.flatten()

        kinter = self.k.to(u.Mpc**-1)
        f_mom = np.power(
            np.reshape(
                self.cosmology.growth_rate(kinter, z, tracer=tracer),
                (*kinter.shape, *z.shape),
            ),
            moment,
        )
        Dkinter = (
            (
                4
                * np.pi
                * (kinter[:, None] / (2 * np.pi)) ** 3
                * np.reshape(
                    self.cosmology.matpow(kinter, z, nonlinear=False, tracer=tracer),
                    (*kinter.shape, *z.shape),
                )
                * f_mom
            )
            .to(1)
            .value
        )
        sigma = np.empty((*R.shape, *z.shape))
        for iR, Ri in enumerate(R):
            for iz, zi in enumerate(z):
                r = np.squeeze(Ri).item()
                sigma[iR, iz] = adaptive_mesh_integral(
                    0,
                    1,
                    ingrnd,
                    args=(r,
                    kinter.value,
                    Dkinter[:, iz],
                    self.haloparams["alpha_iSigma"],
                    ),
                    eps=self.haloparams["tol_sigma"],
                )
        return np.squeeze(np.reshape(sigma, (*Rs, *zs)))

    def read_sigma_lut(self, R, z, output="sigma"):
        """Method to read the sigmaR, dsigmaR look up tabels.
        Will do power law extrapolation.
        Returns dictionarry with desired output with combined shape of the inputs.
        """
        # Array preperation. We will do power law extrapolation in R[Mpc].
        Rs = np.atleast_1d(R).shape
        zs = np.atleast_1d(z).shape
        logR = np.log(np.atleast_1d(R).flatten().to(u.Mpc).value)
        z = np.atleast_1d(z).flatten().astype(float)
        vlogR = np.repeat(logR, len(z))
        vz = np.tile(z, len(logR))

        mlogR = np.log(self.R.to(u.Mpc).value)
        mz = self.z
        result = dict()
        if "sigma" in output or "both" in output:
            sigma = np.exp(
                bilinear_interpolate(mlogR, mz, np.log(self.sigmaR_lut), vlogR, vz)
            )
            result["sigma"] = np.reshape(sigma, (*Rs, *zs))

        if "dsigma" in output or "both" in output:
            dsigma = (
                -np.exp(
                    bilinear_interpolate(
                        mlogR, mz, np.log(-self.dsigmaR_lut), vlogR, vz
                    )
                )
                * u.Mpc**-1
            )
            result["dsigma"] = np.reshape(dsigma, (*Rs, *zs))
        return result

    def sigmaR_of_z(self, R, z, tracer="matter"):
        """Real space variance of matter or cb field smoothed over a comoving scale R"""
        if tracer == self.tracer:
            sigma = self.read_sigma_lut(R, z, output="sigma")["sigma"]
            return np.squeeze(sigma)
        else:
            return np.sqrt(
                self._compute_sigma(R, z, sigma_integrand, tracer=tracer, moment=0.0)
            )

    def sigma8_of_z(self, z, tracer="matter"):
        """Cosmological quantity known as sigma8.
        Will have a slight missmatch with the input sigma8 because
        it is computed numerically from Pk
        """
        return self.sigmaR_of_z(8 * self.Mpch, z, tracer=tracer)

    def fsigma8_of_z(self, k, z, tracer="matter"):
        """(scale-independent) growthrate times sigma8"""
        f = np.reshape(
            self.cosmology.growth_rate(k, z, tracer=tracer),
            (*k.shape, *z.shape),
        )
        s8 = np.reshape(self.sigma8_of_z(z, tracer=tracer), z.shape)
        return f * s8[None, :]

    def dsigmaR_of_z(self, R, z, tracer="matter"):
        """Derivative of real space variance of matter or cb smoothed over a comoving scale R"""
        if tracer == self.tracer:
            dsigma = self.read_sigma_lut(R, z, output="dsigma")["dsigma"]
            return np.squeeze(dsigma)
        else:
            sig = np.sqrt(
                self._compute_sigma(R, z, sigma_integrand, tracer=tracer, moment=0.0)
            )
            dsig = self._compute_sigma(
                R, z, dsigma_integrand, tracer=tracer, moment=0.0
            )
            return dsig / (2 * sig) * u.Mpc**-1

    def n_eff_of_z(self, R, z, tracer="matter"):
        """Effective slope of the power spectrum on comoving scale R"""
        R = np.atleast_1d(R)
        if tracer == self.tracer:
            sigma_dic = self.read_sigma_lut(R, z, output="both")
            n_eff = -2 * R[:, None] * sigma_dic["dsigma"] / sigma_dic["sigma"] - 3.0
        else:
            sig = self._compute_sigma(R, z, sigma_integrand, tracer=tracer, moment=0.0)
            dsig = self._compute_sigma(
                R, z, dsigma_integrand, tracer=tracer, moment=0.0
            )
            n_eff = -R[:, None] * dsig / sig - 3.0

        ns = self.cosmology.cosmopars["ns"]
        n_eff[n_eff < ns - 4] = ns - 4
        n_eff[n_eff > ns] = ns
        return n_eff

    def sigmaV_of_z(self, z, tracer="matter", moment=0):
        """Real space variance of velocity dispercion field Theta."""
        R = np.atleast_1d(
            0.0 * u.Mpc
        )  # Only appears like this in the code but could be easily added as an option
        sv = np.sqrt(
            self._compute_sigma(R, z, sigmav_integrand, tracer=tracer, moment=moment)
        )
        return sv * u.Mpc

    def sigmaM(self, M, z, tracer="matter"):
        """
        Mass (or CDM+baryon) variance at target redshift as a function of collapsed Mass
        """
        rhoM = self.rho_crit * self.cosmology.Omega(0, tracer)
        R = (3.0 * M / (4.0 * np.pi * rhoM)) ** (1.0 / 3.0)

        return self.sigmaR_of_z(R, z, tracer)

    def dsigmaM_dM(self, M, z, tracer="matter"):
        """
        Matter (or CDM+baryon) derivative variance at target redshift as a function of collapsed Mass
        """
        M = np.atleast_1d(M)
        z = np.atleast_1d(z)
        rhoM = self.rho_crit * self.cosmology.Omega(0, tracer)
        R = (3.0 * M / (4.0 * np.pi * rhoM)) ** (1.0 / 3.0)

        np.reshape(self.dsigmaR_of_z(R, z, tracer), (*M.shape, *z.shape)).unit
        dsigma = (
            np.reshape(self.dsigmaR_of_z(R, z, tracer), (*M.shape, *z.shape))
            * (R / (3 * M))[:, None]
        )
        return np.squeeze(dsigma)

    def mass_non_linear(self, z, delta_crit=1.686):
        """
        Get (roughly) the mass corresponding to the nonlinear scale in units of Msun h
        """
        sigmaM_z = self.sigmaM(self.M, z, self.tracer)
        mass_non_linear = self.M[np.argmin(np.power(sigmaM_z - delta_crit, 2), axis=0)]

        return mass_non_linear.to(self.Msunh)

    def sigmav_broadening(self, M, z):
        """Computes the physical scale of the line broadening due to
        galactic rotation curves.
        """
        Mvec = np.atleast_1d(M)[:, None]
        zvec = np.atleast_1d(z)[None, :]

        vM = np.atleast_1d(self.haloparams["v_of_M"](Mvec))
        Hz = np.atleast_1d(self.cosmology.Hubble(zvec))
        sv = vM / self.cosmology.CELERITAS * (1 + zvec) / Hz / np.sqrt(8 * np.log(2))

        return np.squeeze(sv)

    def broadening_FT(self, k, mu, M, z):
        """Fourierspace factor to consider models with line broadening
        Code by Dongwoo T. Chung
        """
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        M = np.atleast_1d(M)
        z = np.atleast_1d(z)

        sv = np.reshape(self.sigmav_broadening(M, z), (1, 1, *M.shape, *z.shape))
        fac = np.exp(
            -1 / 2 * np.power(k[:, None, None, None] * mu[None, :, None, None] * sv, 2)
        )
        return np.squeeze(fac)

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

        rho_input = self.rho_tracer

        dndM = np.reshape(self._dn_dM_of_M(M, rho_input, z), (*M.shape, *z.shape))

        if "f_NL" in self.cosmology.fullcosmoparams:
            Delta_HMF = np.reshape(self.Delta_HMF(M, z), (*M.shape, *z.shape))
            dndM *= 1 + Delta_HMF

        return np.squeeze(dndM).to(u.Mpc**-3 * u.Msun**-1)

    def concentration(self, M, z):
        """Halo concentration red from look up table using log extrapolation in M"""
        M = np.atleast_1d(M)
        z = np.atleast_1d(z)

        kappa = 0.42
        a0 = 2.37
        a1 = 1.74
        b0 = 3.39
        b1 = 1.82
        ca = 0.2

        rhoM = self.rho_crit * self.cosmology.Omega(0, self.tracer)
        R = (3.0 * M / (4.0 * np.pi * rhoM)) ** (1.0 / 3.0)
        R = R.to(u.Mpc)

        nu = self.delta_crit / np.reshape(
            self.sigmaM(M, z, tracer="clustering"),
            (*M.shape, *z.shape),
        )
        neff = np.reshape(
            self.n_eff_of_z(kappa * R, z, tracer=self.tracer),
            (*M.shape, *z.shape),
        )
        alpha_eff = np.reshape(self.cosmology.growth_rate(1e-4 / u.Mpc, z), z.shape)

        # Quantities for c
        A = a0 * (1.0 + a1 * (neff + 3))
        B = b0 * (1.0 + b1 * (neff + 3))
        C = 1.0 - ca * (1.0 - alpha_eff)
        rhs = np.log10(A / nu * (1.0 + nu**2 / B))

        cbar = np.empty((*M.shape, *z.shape))
        for iz, zi in enumerate(z):
            rhs_ = rhs[:, iz]
            ns_ = neff[:, iz]
            cbar[:, iz] = np.power(10,
                bilinear_interpolate(
                    self.conc_G_lut,
                    self.conc_n_lut,
                    self.conc_logc_lut,
                    rhs_,
                    ns_,
                )
            )
        c = cbar * C
        return np.squeeze(c)

    def concentration_Bullock(self, M, z):
        M = np.atleast_1d(M)
        z = np.atleast_1d(z)

        growthnu = self.delta_crit / np.reshape(
            self.sigmaM(0.01 * M, z, tracer=self.tracer),
            (*M.shape, *z.shape)
        ) * self.cosmology.growth_factor(1e-3*u.Mpc**-1, z, tracer=self.tracer)
        gs = growthnu.shape
        growthnu = growthnu.flatten()

        growth = self.cosmology.growth_factor(1e-3*u.Mpc**-1, self.z, tracer=self.tracer)
        zf = np.reshape(linear_interpolate(growth, self.z, growthnu), gs)
        for im in range(len(M)):
            zf[im, :] = np.maximum(zf[im, :], z)
        c = 5.196 * (1 + zf) / (1 + z[None, :])
        return np.squeeze(c)

    def ft_NFW(self, k, M, z):
        """
        Fourier transform of NFW profile, for computing one-halo term
        """
        k = np.atleast_1d(k)
        M = np.atleast_1d(M)
        z = np.atleast_1d(z)

        # Radii of the SO collapsed (assuming 200*rho_crit)
        Delta = 200.0
        rho_crit = self.rho_crit
        R_NFW = (3.0 * M / (4.0 * np.pi * Delta * rho_crit)) ** (1.0 / 3.0)

        # get characteristic radius
        if self.haloparams["concentration"] == "Bullock01":
            c = np.reshape(self.concentration_Bullock(M, z), (*M.shape, *z.shape))[None, :, :]
        elif self.haloparams["concentration"] == "Diemer19":
            c = np.reshape(self.concentration(M, z), (*M.shape, *z.shape))[None, :, :]
        else:
            raise ValueError("Concentraion relation not found")
        r_s = R_NFW[None, :, None] / c
        gc = np.log(1 + c) - c / (1.0 + c)
        # argument: k*rs
        x = (k[:, None, None] * r_s).to(1).value

        if self.haloparams["bloating"] == "Mead20":
            nu  = self.delta_crit / np.reshape(
                self.sigmaM(M, z, tracer=self.tracer),
                (*M.shape, *z.shape)
            )
            eta = 0.1281 * np.reshape(
                self.sigma8_of_z(z, tracer=self.tracer), z.shape,
            )**-0.3644
            x = x * nu[None, :, :]**eta[None, None, :]

        si_x, ci_x = sici(x)
        si_cx, ci_cx = sici((1.0 + c) * x)
        u_km = (
            np.cos(x) * (ci_cx - ci_x)
            + np.sin(x) * (si_cx - si_x)
            - np.sin(c * x) / ((1.0 + c) * x)
        )
        return np.squeeze(u_km / gc)

    def neff_NL(self, z, delta_crit=1.686):
        M_NL = self.mass_non_linear(z, delta_crit=delta_crit)
        R_NL = np.power((3 * M_NL) / (4 * np.pi * self.rho_tracer), 1 / 3)
        return self.n_eff_of_z(R_NL, z, tracer=self.tracer)

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
        tracer = self.tracer
        M = np.atleast_1d(M)
        z = np.atleast_1d(z)

        rhoM = self.rho_tracer
        R = (3.0 * M / (4.0 * np.pi * rhoM)) ** (1.0 / 3.0)
        kmin = cfg.settings["kmin"]
        kmax = cfg.settings["kmax"]
        k = self.k
        mu = np.linspace(-0.995, 0.995, 128)

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
        S3, dS3_dM = self.S3_dS3(M, z, self.tracer)
        sigmaM = self.sigmaM(M, z, self.tracer)
        dSigmaM = self.dsigmaM_dM(M, z, self.tracer)

        kappa3 = S3 / sigmaM
        dkappa3dM = (dS3_dM - 3 * S3 * dSigmaM / sigmaM) / (sigmaM**3)

        return kappa3, dkappa3dM

    def Delta_HMF(self, M, z):
        """
        The correction to the HMF due to non-zero f_NL, as presented in 2009.01245.
        """
        sigmaM = self.sigmaM(M, z, self.tracer)
        dSigmaM = self.dsigmaM_dM(M, z, self.tracer)

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
        tracer = self.tracer

        M = np.atleast_1d(M)
        z = np.atleast_1d(z)
        k = np.atleast_1d(k)

        Tk = np.reshape(self.cosmology.Transfer(k, z), (*k.shape, *z.shape))
        bias = np.reshape(self._b_of_M(M, z, self.delta_crit), (*M.shape, *z.shape))

        f1 = (
            (self.cosmology.Hubble(0, physical=True) / (self.cosmology.CELERITAS * k))
            .to(1)
            .value
        )
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

    ##################################
    # real space halo power spectrum #
    ##################################

    def Ihalo(self, z, bstring, *args, p=1, scale=()):
        M = self.M.to(u.Msun)
        z = np.atleast_1d(z)

        if len(args) % p != 0:
            raise ValueError("You have to pass wave-vectors for every p")
        else:
            kd = [np.atleast_1d(args[ik]) for ik in range(p)]

        if scale:
            alpha = np.array([*scale])
        else:
            alpha = np.ones(p)

        # Independent of k
        dndM = np.reshape(self.halomassfunction(M, z), (*M.shape, *z.shape))
        bfunc = getattr(self._bias_function, bstring)
        b = np.reshape(bfunc(M, z, self.delta_crit), (*M.shape, *z.shape))

        # Dependent on k
        normhaloprofile = []
        fac = self.M[:, None] / self.rho_tracer
        for ik in range(p):
            k = kd[ik]
            U = np.reshape(
                self.ft_NFW(k, M, z),
                (*k.shape, *M.shape, *z.shape),
             ) * fac
            U = np.expand_dims(U, (*range(ik), *range(ik + 1, p)))
            U = np.expand_dims(U, (*range(p, 2 * p),))
            normhaloprofile.append(np.power(U, alpha[ik]))

        # Dependent on k and mu
        Fv = np.ones((p,))
        if self.haloparams["v_of_M"]:
            Fv = []
            for ik in range(p):
                k = kd[ik]
                mu = np.atleast_1d(args[p + ik])
                F = np.reshape(
                    self.broadening_FT(k, mu, M, z),
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
                Fv.append(np.power(F, alpha[ik]))

        # Construct the integrand
        Intgrnd = b * dndM * M[:, None]
        for ik in range(p):
            Intgrnd = Intgrnd * Fv[ik] * normhaloprofile[ik]
        logM = np.log(M.value)
        Umean = np.trapz(Intgrnd, logM, axis=-2)
        return np.squeeze(Umean)

    def Ibeta_1(self, k, z, mu=None, beta=0):
        if beta==0:
            bstring = "b0"
        elif beta==1:
            bstring = self.haloparams["bias_model"]
        else:
            bstring = beta
        return self.Ihalo(z, bstring, k, mu, p=1, scale=(1,))

    def Ibeta_2(self, k, z, mu=None, beta=0):
        if beta==0:
            bstring = "b0"
        elif beta==1:
            bstring = self.haloparams["bias_model"]
        else:
            bstring = beta
        return self.Ihalo(z, bstring, k, k, mu, mu, p=2, scale=(1, 1,))

    def Ibeta_3(self, k, z, mu=None, beta=0):
        if beta==0:
            bstring = "b0"
        elif beta==1:
            bstring = self.haloparams["bias_model"]
        else:
            bstring = beta
        return self.Ihalo(z, bstring, k, k, mu, mu, p=2, scale=(2, 1,))

    def Ibeta_4(self, k, z, mu=None, beta=0):
        if beta==0:
            bstring = "b0"
        elif beta==1:
            bstring = self.haloparams["bias_model"]
        else:
            bstring = beta
        return self.Ihalo(z, bstring, k, k, mu, mu, p=2, scale=(2, 2,))

    def P_QNL(self, k, z):
        k = np.atleast_1d(k)
        z = np.atleast_1d(z)

        P = np.reshape(self.cosmology.matpow(k, z, tracer=self.tracer),
                       (*k.shape, *z.shape),
        )
        Pnw = np.reshape(self.cosmology.nonwiggle_pow(k, z, tracer=self.tracer),
                         (*k.shape, *z.shape),
        )
        gd = np.exp(-(k[:, None]* self.sigmaV_of_z(z, tracer="matter", moment=0)).to(1).value)
        return P * gd + Pnw * (1 - gd)

    def P_halo(self, k, z, mu=None):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)

        P_QNL = np.reshape(self.P_QNL(k, z), (*k.shape, *z.shape))

        I1_1 = restore_shape(self.Ibeta_1(k, z, mu=mu, beta=1), k, mu, z)
        I1_norm = restore_shape(self.Ibeta_1(1e-4*u.Mpc**-1, z, mu=mu, beta=1), mu, z)
        I1_1 = I1_1 / I1_norm

        I0_2 = restore_shape(self.Ihalo(z, "b0", k, mu, p=1, scale=(2,)), k, mu, z)

        alpha = 1
        if self.haloparams["transition_smoothing"] == "Mead2020":
            alpha = (1.875 * (1.603)**self.neff_NL(z))[None, None, :]

        D1h = ((k[:, None, None] / (2 * np.pi))**3 * I0_2).to(1).value
        D2h = ((k[:, None, None] / (2 * np.pi))**3 * I1_1**2 * P_QNL[:, None, :]).to(1).value

        P = (D1h**alpha + D2h**alpha)**(1/alpha) * (k[:, None, None] / (2 * np.pi))**-3
        return np.squeeze(P)

##############
# Numba Part #
##############


@njit("(float64, float64, float64[::1], float64[:], uint64)", fastmath=True)
def sigma_integrand(t, R, kinter, Dkinter, alpha):
    # mask out region where integrand is 0
    if t<=0 or t>=1:
        return 0.0

    Rk = (1 / t - 1) ** alpha
    k = Rk / R
    W = smooth_W(np.array([Rk]))[0]

    # power law extrapolation
    Dk = np.exp(
        linear_interpolate(
            np.log(kinter),
            np.log(Dkinter),
            np.log(np.array([k])),
        )
    )[0]
    return alpha * Dk * W**2 / (t * (1 - t))


@njit("(float64, float64, float64[::1], float64[:], uint64)", fastmath=True)
def dsigma_integrand(t, R, kinter, Dkinter, alpha):
    # mask out region where integrand is 0
    if t<=0 or t>=1:
        return 0.0

    Rk = (1 / t - 1) ** alpha
    k = Rk / R
    W = smooth_W(np.array([Rk]))[0]
    dW = smooth_dW(np.array([Rk]))[0]

    # power law extrapolation
    Dk = np.exp(
        linear_interpolate(
            np.log(kinter),
            np.log(Dkinter),
            np.log(np.array([k])),
        )
    )[0]
    return alpha * Dk * 2 * k * dW * W / (t * (1 - t))


@njit("(float64, float64, float64[::1], float64[:], uint64)", fastmath=True)
def sigmav_integrand(t, R, kinter, Dkinter, alpha):
    # mask out region where integrand is 0
    if t<=0 or t>=1:
        return 0.0

    if np.isclose(R, 0):
        Rk = 0
        k = (1 / t - 1) ** alpha  # Don't worry to hard about the units
        W = 1
    else:
        Rk = (1 / t - 1) ** alpha
        k = Rk / R
        W = smooth_W(np.array([Rk]))[0]

    # power law extrapolation
    Dk = np.exp(
        linear_interpolate(
            np.log(kinter),
            np.log(Dkinter),
            np.log(np.array([k])),
        )
    )[0]
    return alpha * Dk * W**2 / (3 * k**2 * t * (1 - t))

@njit(
    "(float64[::1], float64[::1], float64[::1], float64[:,:], uint64, float64)",
    parallel=True,
)
def sigmas_of_R_and_z(R, z, kinter, Dkinter, alpha, eps):
    Rl, zl = len(R), len(z)
    sigma = np.empty((Rl, zl))
    dsigma = np.empty((Rl, zl))

    # integration will be done on a log-spaced grid in k from 0 to infinity via transformation
    for iz in prange(zl):
        for iR in prange(Rl):
            sintegral = adaptive_mesh_integral(
                0, 1, sigma_integrand, args=(R[iR], kinter, Dkinter[:, iz], alpha), eps=eps,
            )
            sigma[iR, iz] = np.sqrt(sintegral)
            dsintegral = adaptive_mesh_integral(
                0, 1, dsigma_integrand, args=(R[iR], kinter, Dkinter[:, iz], alpha), eps=eps,
            )
            dsigma[iR, iz] = dsintegral / (2 * sigma[iR, iz])
    return sigma, dsigma
