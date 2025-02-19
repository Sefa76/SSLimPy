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
        self.concentration_lut = self._create_concentration_lookuptable()

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
        zvec = self.z
        Rvec = self.R
        Mvec = self.M

        # fit parameters
        kappa = 0.42
        a0 = 2.37
        a1 = 1.74
        b0 = 3.39
        b1 = 1.82
        ca = 0.2

        nu = self.delta_crit / self.sigmaM(Mvec, zvec, tracer="clustering")
        alpha_eff = self.cosmology.growth_rate(1e-4 / u.Mpc, zvec)
        neff = self.n_eff_of_z(kappa * Rvec, zvec, tracer=self.tracer)

        # Quantities for c
        A = a0 * (1.0 + a1 * (neff + 3))
        B = b0 * (1.0 + b1 * (neff + 3))
        C = 1.0 - ca * (1.0 - alpha_eff)
        arg = A / nu * (1.0 + nu**2 / B)

        # Compute G(x), with x = r/r_s, and evaluate c
        x = np.logspace(-3, 3, 256)
        g = np.log(1 + x) - x / (1.0 + x)
        G = x[None, None, :] / np.power(
            g[None, None, :], (5.0 + neff[:, :, None]) / 6.0
        )

        c = np.zeros_like(arg)
        for iM, G_z_and_x in enumerate(G):
            for iz, G_x in enumerate(G_z_and_x):
                c[iM, iz] = C[iz] * linear_interpolate(
                    G_x, x, np.atleast_1d(arg[iM, iz])
                )

        return c

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
                    r,
                    kinter.value,
                    Dkinter[:, iz],
                    self.haloparams["alpha_iSigma"],
                    self.haloparams["tol_sigma"],
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
        """Halo concentraion red from look up table using log extrapolation in M"""
        Ms = np.atleast_1d(M).shape
        zs = np.atleast_1d(z).shape
        logM = np.log(np.atleast_1d(M).flatten().to(u.Msun).value)
        z = np.atleast_1d(z).flatten().astype(float)
        vlogM = np.repeat(logM, len(z))
        vz = np.tile(z, len(logM))

        logMinter = self.M.to(u.Msun)
        c = np.reshape(
            bilinear_interpolate(logMinter, self.z, self.concentration_lut, vlogM, vz),
            (*Ms, *zs),
        )
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

    def ft_NFW(self, k, M, z, bloating=False, concB=False):
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
        if concB:
            c = np.reshape(self.concentration_Bullock(M, z), (*M.shape, *z.shape))[None, :, :]
        else:
            c = np.reshape(self.concentration(M, z), (*M.shape, *z.shape))[None, :, :]
        r_s = R_NFW[None, :, None] / c
        gc = np.log(1 + c) - c / (1.0 + c)
        # argument: k*rs
        x = (k[:, None, None] * r_s).to(1).value

        if bloating:
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

    def P1h(self, k, z, mu=None, bloating=False, concB=False):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        M = self.M.to(u.Msun)
        z = np.atleast_1d(z)

        dndM = np.reshape(self.halomassfunction(M, z),
                          (*M.shape, *z.shape),
        )

        fac = self.M[:, None] / self.rho_tracer
        W = (
            np.reshape(
                self.ft_NFW(k, M, z, bloating=bloating, concB=concB),
                (*k.shape, *M.shape, *z.shape),
             ) * fac
        ).to(u.Mpc**3)

        Fv = 1
        if self.haloparams["v_of_M"]:
            Fv = np.reshape(
                self.broadening_FT(k, mu, M, z),
                (*k.shape, *mu.shape, *M.shape, *z.shape)
            )
        Intgrnd = (
            dndM[None, None, :, :]
            * W[:, None, :, :]**2
            * Fv**2
        )
        P1h = np.trapz(Intgrnd * M[None, None, :, None], np.log(M.value), axis=-2)
        return P1h

    def PQnl(self, k, z):
        k = np.atleast_1d(k)
        z = np.atleast_1d(z)

        P = np.reshape(self.cosmology.matpow(k, z, tracer="clustering"),
                       (*k.shape, *z.shape),
        )
        Pnw = np.reshape(self.cosmology.nonwiggle_pow(k, z, tracer="clustering"),
                         (*k.shape, *z.shape),
        )
        gd = np.exp(-(k[:, None]* self.sigmaV_of_z(z, tracer="matter", moment=0)).to(1).value)
        return P * gd + Pnw * (1 - gd)
    
    def Cclust(self, k, z, mu=None):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        M = self.M.to(u.Msun)
        z = np.atleast_1d(z)

        dndM = np.reshape(self.halomassfunction(M, z),
                          (*M.shape, *z.shape),
        )

        fac = self.M[:, None] / self.rho_tracer
        W = (
            np.reshape(
                self.ft_NFW(k, M, z),
                (*k.shape, *M.shape, *z.shape),
             ) * fac
        ).to(u.Mpc**3)      

        Fv = 1
        if self.haloparams["v_of_M"]:
            Fv = np.reshape(
                self.broadening_FT(k, mu, M, z),
                (*k.shape, *mu.shape, *M.shape, *z.shape)
            )
        
        b = restore_shape(self.halobias(M, z, k=k), k, M, z)
        
        Intgrnd = (
            dndM[None, None, :, :]
            * W[:, None, :, :]
            * Fv
            * b[:, None, :, :]
        )
        clust = np.trapz(Intgrnd * M[None, None, :, None], np.log(M.value), axis=-2)
        return np.squeeze(clust)

    def P_halo(self, k, z, mu=None, bloating=False, P2h_sup= False, concB=False):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)

        P1h = restore_shape(self.P1h(k, z, mu=mu, bloating=bloating, concB=concB), k, mu, z)
        # Cclust = restore_shape(self.Cclust(k, z, mu=mu), k, mu, z)
        Cclust = 1
        if P2h_sup:
            nd = 2.853
            f = np.reshape(
                0.2696 * self.sigma8_of_z(z, tracer=self.tracer)**0.9403, z.shape
            )
            kd = np.reshape(
                0.05699 * self.Mpch**-1 * self.sigma8_of_z(z, tracer=self.tracer)**-1.089,
                z.shape
            )
            x = ((k[:,None]/kd[None,:]).to(1).value)**nd
            Cclust = Cclust * (1- f* x/(1+x))[:, None, :]

        PQnl = np.reshape(self.PQnl(k, z), (*k.shape, *z.shape))
        P2h = PQnl[:, None, :] * Cclust

        M_nl = self.mass_non_linear(z)
        R_nl = (3.0 * M_nl / (4.0 * np.pi * self.rho_tracer)) ** (1.0 / 3.0)
        alpha_smooth = np.reshape(1.875 * (1.603)**self.n_eff_of_z(R_nl, z, tracer="clustering"), z.shape)

        PtD = 4 * np.pi* (k / (2 * np.pi))**3
        PtD = PtD[:, None, None]
        P = 1 / PtD * ((PtD * P2h)**alpha_smooth + (PtD * P1h)**alpha_smooth)**(1/alpha_smooth) 

        return np.squeeze(P)

##############
# Numba Part #
##############


@njit("(float64[::1], float64, float64[::1], float64[:], uint64)", fastmath=True)
def sigma_integrand(t, R, kinter, Dkinter, alpha):
    nt = len(t)
    integrand = np.zeros(nt)
    # mask out region where integrand is 0
    mask = (t > 0) & (t < 1)

    Rk = (1 / t[mask] - 1) ** alpha
    k = Rk / R
    W = smooth_W(Rk)

    # power law extrapolation
    Dk = np.exp(
        linear_interpolate(
            np.log(kinter),
            np.log(Dkinter),
            np.log(k),
        )
    )
    integrand[mask] = alpha * Dk * W**2 / (t[mask] * (1 - t[mask]))
    return integrand


@njit("(float64[::1], float64, float64[::1], float64[:], uint64)", fastmath=True)
def dsigma_integrand(t, R, kinter, Dkinter, alpha):
    nt = len(t)
    integrand = np.zeros(nt)
    # mask out region where integrand is 0
    mask = (t > 0) & (t < 1)
    Rk = (1 / t[mask] - 1) ** alpha
    k = Rk / R
    W = smooth_W(Rk)
    dW = smooth_dW(Rk)

    # power law extrapolation
    Dk = np.exp(
        linear_interpolate(
            np.log(kinter),
            np.log(Dkinter),
            np.log(k),
        )
    )
    integrand[mask] = alpha * Dk * 2 * k * dW * W / (t[mask] * (1 - t[mask]))
    return integrand


@njit("(float64[::1], float64, float64[::1], float64[:], uint64)", fastmath=True)
def sigmav_integrand(t, R, kinter, Dkinter, alpha):
    nt = len(t)
    integrand = np.zeros(nt)
    # mask out region where integrand is 0
    mask = (t > 0) & (t < 1)

    if np.isclose(R, 0):
        Rk = 0
        k = (1 / t[mask] - 1) ** alpha  # Don't worry to hard about the units
        W = np.ones_like(k)
    else:
        Rk = (1 / t[mask] - 1) ** alpha
        k = Rk / R
        W = smooth_W(Rk)

    # power law extrapolation
    Dk = np.exp(
        linear_interpolate(
            np.log(kinter),
            np.log(Dkinter),
            np.log(k),
        )
    )
    integrand[mask] = alpha * Dk * W**2 / (3 * k**2 * t[mask] * (1 - t[mask]))
    return integrand


@njit
def adaptive_mesh_integral(a, b, integrand, R, kinter, Dkinter, alpha, eps):
    """Adapted from CAMB and Class implementation of HMCode2020 by Alexander Mead"""
    if a == b:
        return 0

    # Define the minimum and maximum number of iterations
    jmin = 5  # Minimum iterations to avoid premature convergence
    jmax = 20  # Maximum iterations before timeout

    # Initialize sum variables for integration
    sum_2n = 0.0
    sum_n = 0.0
    sum_old = 0.0
    sum_new = 0.0

    for j in range(1, jmax + 1):
        n = 1 + 2 ** (j - 1)
        dx = (b - a) / (n - 1)
        if j == 1:
            f1 = integrand(np.array([float(a)]), R, kinter, Dkinter, alpha)[0]
            f2 = integrand(np.array([float(b)]), R, kinter, Dkinter, alpha)[0]
            # print(f1, f2)
            sum_2n = 1.0 / 2.0 * (f1 + f2) * dx
            sum_new = sum_2n
        else:
            t = a + (b - a) * (np.arange(2, n, 2) - 1) / (n - 1)
            I = integrand(t, R, kinter, Dkinter, alpha)
            sum_2n = sum_n / 2 + np.sum(I) * dx
            sum_new = (4 * sum_2n - sum_n) / 3
            # print(sum_new, sum_old)

        if j >= jmin and sum_old != 0:
            if abs(1.0 - sum_new / sum_old) < eps:
                return sum_new
        if j == jmax:
            print("INTEGRATE: Integration timed out")
            return sum_new
        else:
            sum_old = sum_new
            sum_n = sum_2n
            sum_2n = 0.0
    return sum_new


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
                0, 1, sigma_integrand, R[iR], kinter, Dkinter[:, iz], alpha, eps
            )
            sigma[iR, iz] = np.sqrt(sintegral)
            dsintegral = adaptive_mesh_integral(
                0, 1, dsigma_integrand, R[iR], kinter, Dkinter[:, iz], alpha, eps
            )
            dsigma[iR, iz] = dsintegral / (2 * sigma[iR, iz])
    return sigma, dsigma
