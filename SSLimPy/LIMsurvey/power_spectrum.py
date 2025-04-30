from copy import copy
from time import time
import numpy as np
from astropy import units as u
from numba import njit, prange
from scipy.integrate import simpson
from scipy.interpolate import RectBivariateSpline
from scipy.special import legendre
from SSLimPy.cosmology.astro import AstroFunctions
from SSLimPy.interface import config as cfg
from SSLimPy.utils.utils import *


class PowerSpectra:

    def __init__(self, astro: AstroFunctions, BAOpars=dict(), settings=dict()):
        self.cosmology = astro.cosmology
        self.halomodel = astro.halomodel
        # Nu enters into the Mass-Luminosity and Luminosity-Temperature relations
        self.survey_specs = astro.survey_specs 
        self.astro = astro

        self.BAOpars = copy(BAOpars)
        self.tracer = self.halomodel.tracer

        self.fiducial_cosmology = cfg.fiducialcosmo
        self.fiducial_halomodel = cfg.fiducialhalomodel

        #################################
        # Properties of target redshift #
        #################################

        self.nu = astro.nu
        self.nuObs = astro.nuObs
        self.z = np.sort(np.atleast_1d(self.nu / self.nuObs).to(1).value - 1)

        #########################################
        # Masses, luminosities, and wavenumbers #
        #########################################
        self.M = astro.M
        self.L = astro.L

        self.set_survey_kspace(settings)

        self.compute_power_spectra()
        self.compute_power_spectra_moments()

    def set_survey_kspace(self, settings):
        """Constructs the kspace for the survey
        Defaults back to config if not passed
        """
        k_kind = settings.get("k_kind", cfg.settings["k_kind"])
        nk = settings.get("nk", cfg.settings["nk"])
        kmin = settings.get("kmin", cfg.settings["kmin"])
        kmax = settings.get("kmax", cfg.settings["kmax"])

        if k_kind == "log":
            k_edge = np.geomspace(kmin, kmax, nk).to(u.Mpc**-1)
        else:
            k_edge = np.linspace(kmin, kmax, nk).to(u.Mpc**-1)
        self.k = (k_edge[1:] + k_edge[:-1]) / 2.0
        self.dk = np.diff(k_edge)
        self.k_numerics = self.halomodel.k

        nmu = settings.get("nmu", 128)

        mu_edge = np.linspace(-1, 1, nmu + 1)
        self.mu = (mu_edge[:-1] + mu_edge[1:]) / 2.0
        self.dmu = np.diff(mu_edge)

    ###############
    # De-Wiggling #
    ###############

    def sigmavNL(self, mu, z, BAOpars=dict()):
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)

        if cfg.settings["fix_cosmo_nl_terms"]:
            cosmo = self.fiducial_cosmology
            halomodel = self.fiducial_halomodel
        else:
            cosmo = self.cosmology
            halomodel = self.halomodel

        if "sigmav" in BAOpars:
            sigmav = np.atleast_1d(BAOpars["sigmav"])
            if len(sigmav) != len(z):
                raise ValueError(
                    "did not pass velocity dispertion for every z asked for"
                )
            # scale independent f
            f_scaleindependent = np.atleast_1d(cosmo.growth_rate(1e-3 / u.Mpc, z))
            sv2 = np.power(sigmav[None, :], 2) * (
                1
                - np.power(mu, 2)[:, None]
                + np.power(mu, 2)[:, None] * (1 + f_scaleindependent[None, :]) ** 2
            )
        else:
            f0 = np.atleast_1d(np.power(halomodel.sigmaV_of_z(z, moment=0), 2))[:, None]
            f1 = np.atleast_1d(np.power(halomodel.sigmaV_of_z(z, moment=1), 2))[:, None]
            f2 = np.atleast_1d(np.power(halomodel.sigmaV_of_z(z, moment=2), 2))[:, None]
            sv2 = f0 + 2 * mu[None, :] ** 2 * f1 + mu[None, :] ** 2 * f2
        return np.squeeze(np.sqrt(sv2))

    def dewiggled_pdd(self, k, mu, z, BAOpars=dict()):
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

        gmudamping = np.reshape(
            self.sigmavNL(mu, z, BAOpars=BAOpars) ** 2, (*mu.shape, *z.shape)
        )

        P_dd = self.cosmology.matpow(k, z, tracer=self.tracer, nonlinear=False)
        P_dd_NW = self.cosmology.nonwiggle_pow(
            k, z, tracer=self.tracer, nonlinear=False
        )
        P_dd = np.reshape(P_dd, (*k.shape, *z.shape))
        P_dd_NW = np.reshape(P_dd_NW, (*k.shape, *z.shape))

        P_dd_DW = P_dd[:, None, :] * np.exp(
            -gmudamping[None, :, :] * k[:, None, None] ** 2
        ) + P_dd_NW[:, None, :] * (
            1 - np.exp(-gmudamping[None, :, :] * k[:, None, None] ** 2)
        )
        return np.squeeze(P_dd_DW)

    ################
    # BAO Features #
    ################

    def alpha_parallel(self, z, BAOpars=dict()):
        """
        Function implementing alpha_parallel of the Alcock-Paczynski effect
        If BAOpars is passed checks for alpha_iso and alpha_AP
        """
        z = np.atleast_1d(z)
        if "alpha_iso" in BAOpars:
            alpha_iso = np.atleast_1d(BAOpars["alpha_iso"])
            if len(z) != len(alpha_iso):
                raise ValueError(
                    "did not pass alpha_iso parameters for every z asked for"
                )
            if "alpha_AP" in BAOpars:
                alpha_AP = np.atleast_1d(BAOpars["alpha_AP"])
                if len(z) != len(alpha_AP):
                    raise ValueError(
                        "did not pass alpha_AP parameters for every z asked for"
                    )
            else:
                alpha_AP = 1.0
            alpha_par = alpha_iso * np.power(alpha_AP, 2 / 3)
        else:
            fidTerm = self.fiducial_cosmology.Hubble(z)
            cosmoTerm = self.cosmology.Hubble(z)
            alpha_par = fidTerm / cosmoTerm * self.dragscale()
        return alpha_par

    def alpha_perpendicular(self, z, BAOpars=dict()):
        """
        Function implementing alpha_perpendicular of the Alcock-Paczynski effect
        If BAOpars is passed checks for alpha_iso and alpha_AP
        """
        z = np.atleast_1d(z)
        if "alpha_iso" in BAOpars:
            alpha_iso = np.atleast_1d(BAOpars["alpha_iso"])
            if len(z) != len(alpha_iso):
                raise ValueError(
                    "did not pass alpha_iso parameters for every z asked for"
                )
            if "alpha_AP" in BAOpars:
                alpha_AP = np.atleast_1d(BAOpars["alpha_AP"])
                if len(z) != len(alpha_AP):
                    raise ValueError(
                        "did not pass alpha_AP parameters for every z asked for"
                    )
            else:
                alpha_AP = 1.0
            alpha_perp = alpha_iso / np.power(alpha_AP, 1 / 3)
        else:
            fidTerm = self.fiducial_cosmology.angdist(z)
            cosmoTerm = self.cosmology.angdist(z)
            alpha_perp = cosmoTerm / fidTerm * self.dragscale()
        return alpha_perp

    def dragscale(self):
        """
        Function to fix the dragscale
        """
        fidTerm = self.fiducial_cosmology.rs_drag()
        cosmoTerm = self.cosmology.rs_drag()
        return fidTerm / cosmoTerm

    def bias_term(self, z, k=None, mu=None, BAOpars=dict()):
        """
        Function to compute the bias term that enters the linear Kaiser formula
        If BAOpars is passed checks for bmean, Tmean
        """
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)
        if "bmean" in BAOpars:
            bmean = np.atleast_1d(BAOpars["bmean"])
            if len(bmean) != len(z):
                raise ValueError("did not pass mean bias for every z asked for")

            # Does not contain correction for f_nl yet
            bmean = bmean[None, :]
            if "f_NL" in self.cosmology.fullcosmoparams:
                M_pivot = 1e7 * u.Msun
                fac = (bmean - 1) / np.reshape(
                    self.halomodel._b_of_M(M_pivot, z) - 1, z.shape
                )
                Delta_b = self.halomodel.Delta_b(k, M_pivot, z) * fac[None, :]

                bmean = bmean + Delta_b

            if "Tmean" in BAOpars:
                Tmean = np.atleast_1d(BAOpars["Tmean"])
                if len(Tmean) != len(z):
                    raise ValueError(
                        "did not pass mean Temperature for every z asked for"
                    )
                Tmean = Tmean[None, None, :]
            else:
                Tmean = restore_shape(self.astro.Thalo(z, k, mu, p=1), k, mu, z)

            Biasterm = (
                bmean[:, None, :]
                * Tmean
                * np.atleast_1d(self.halomodel.sigma8_of_z(z, tracer=self.tracer))[
                    None, None, :
                ]
            )
        else:
            Biasterm = (
                restore_shape(self.astro.bhalo(k, z, mu=mu), k, mu, z)
                * np.reshape(self.halomodel.sigma8_of_z(z, tracer=self.tracer), z.shape)[None, None, :]
            )
        return np.squeeze(Biasterm)

    def f_term(self, k, mu, z, BAOpars=dict()):
        """
        Function to compute the linear redshift space distortion that enters the linear Kaiser formula
        If BAOpars is passed checks for Tmean
        """
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)
        if "Tmean" in BAOpars:
            Tmean = np.atleast_1d(BAOpars["Tmean"])
            if len(Tmean) != len(z):
                raise ValueError("did not pass mean Temperature for every z asked for")
            Tmean = Tmean[None, None, :]
        else:
            Tmean = restore_shape(self.astro.Thalo(z, k, mu, p=1), k, mu, z)

        fs8 = np.reshape(
            self.halomodel.fsigma8_of_z(k, z, tracer=self.tracer), (*k.shape, *z.shape)
        )
        Kaiser_RSD = Tmean * fs8[:, None, :] * np.power(mu, 2)[None, :, None]
        return np.squeeze(Kaiser_RSD)

    def Kaiser_Term(self, k, mu, z, BAOpars=dict()):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)
        bterm = restore_shape(self.bias_term(z, k=k, mu=mu, BAOpars=BAOpars), k, mu, z)
        if "beta" in BAOpars:
            beta = np.atleast_1d(BAOpars["beta"])
            if len(beta) != len(z):
                raise ValueError("did not pass RSD amplitude for every z asked for")
            fterm = beta[None, None, :] * np.power(mu[None, :, None], 2)
            linear_Kaiser = np.power(bterm * (1 + fterm), 2)
        else:
            fterm = np.reshape(
                self.f_term(k, mu, z, BAOpars=BAOpars),
                (*k.shape, *mu.shape, *z.shape),
            )
            linear_Kaiser = np.power(bterm + fterm, 2)
        return np.squeeze(linear_Kaiser)

    def fingers_of_god(self, k, mu, z, BAOpars=dict()):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)

        if cfg.settings["fix_cosmo_nl_terms"]:
            cosmo = self.fiducial_cosmology
            halomodel = self.fiducial_halomodel
        else:
            cosmo = self.cosmology
            halomodel = self.halomodel

        if "sigmap" in BAOpars:
            # This quantitiy is different from the old lim one by a factor of f^2
            sigmap = np.atleast_1d(BAOpars["sigmap"])
            if len(sigmap) != len(z):
                raise ValueError(
                    "did not pass velocity dispertion for every z asked for"
                )
            # scale independent f
            f_scaleindependent = cosmo.growth_rate(1e-3 / u.Mpc, z)
            sp = sigmap * f_scaleindependent
        else:
            sp = np.atleast_1d(halomodel.sigmaV_of_z(z, moment=2))
        FoG_damp = cfg.settings["FoG_damp"]
        if FoG_damp == "Lorentzian":
            FoG = np.power(
                1.0
                + 0.5
                * np.power(k[:, None, None] * mu[None, :, None] * sp[None, None, :], 2),
                -2,
            )
        elif FoG_damp == "Gaussian":
            FoG = np.exp(
                -((k[:, None, None] * mu[None, :, None] * sp[None, None, :]) ** 2.0)
            )
        elif FoG_damp == "ISTF_like":
            FoG = np.power(
                1
                + np.power(k[:, None, None] * mu[None, :, None] * sp[None, None, :], 2),
                -1,
            )

        return np.squeeze(FoG)

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
        Wsurvey, Vsurvey = self.survey_specs.Wsurvey(q, muq)

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
        self.Pk_Obs = Pconv * Pobs.unit / Vsurvey.value
        return self.Pk_Obs

    #######################
    # Power Spectra Terms #
    #######################

    def shotnoise(self, z, k=None, mu=None, BAOpars=dict()):
        """
        This function returns additional contributions to the Auto-power spectrum depending on what is asked for:
            - The Posoinian Shot noise
            - The One-Halo Term as scale dependent shot noise
            - Additional Shot noise from the BAOpars
        """
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        z = np.atleast_1d(z)

        Ps = 0
        if "Pshot" in BAOpars:
            Pshot = np.atleast_1d(BAOpars["Pshot"])
            if len(Pshot) != len(z):
                raise ValueError("did not pass the shotnoise for every z asked for")
            Ps = Pshot[None, None, :]
        else:
            if cfg.settings["halo_model_PS"]:
                Ps = self.astro.T_one_halo(k, z, mu=mu)
            else:
                Ps = self.astro.Tavg(z, p=2)
        return np.squeeze(Ps)

    def compute_power_spectra(self):
        """This function computes the full shape observed power spectrum
        with (nearly) fully vectorized function
        """
        # Compute everything on grid quantities
        k = copy(self.k_numerics)
        mu = copy(self.mu)
        z = copy(self.z)
        gridshape = (*k.shape, *mu.shape, *z.shape)
        if cfg.settings["verbosity"] > 1:
            print("requested Pk shape:", gridshape)
            tstart = time()

        if cfg.settings["QNLpowerspectrum"]:
            # Obtain the normalized dewiggled power spectrum
            Pk = np.reshape(
                self.dewiggled_pdd(k, mu, z, BAOpars=self.BAOpars), gridshape
            )
        else:
            # Use linear power spectrum
            Pk = np.reshape(
                self.cosmology.matpow(k, z, tracer=self.tracer), (*k.shape, *z.shape)
            )
            Pk = Pk[:, None, :]

        if cfg.settings["verbosity"] > 1:
            tPk = time()
            print("Power spectrum obtained in {} seconds".format(tPk - tstart))

        if cfg.settings["do_RSD"]:
            # Obtain redshiftspace distortions
            rsd = restore_shape(
                self.Kaiser_Term(k, mu, z, BAOpars=self.BAOpars),
                k,
                mu,
                z,
            )
            if cfg.settings["nonlinearRSD"]:
                rsd = rsd * np.reshape(
                    self.fingers_of_god(k, mu, z, BAOpars=self.BAOpars),
                    (*k.shape, *mu.shape, *z.shape),
                )
        else:
            rsd = restore_shape(
                self.bias_term(z, k=k, mu=mu, BAOpars=self.BAOpars),
                k,
                mu,
                z,
            )

        if cfg.settings["verbosity"] > 1:
            trsd = time()
            print(
                "Redshift space distortions obtained in {} seconds".format(trsd - tPk)
            )

        Ps = restore_shape(
            self.shotnoise(z, k=k, mu=mu, BAOpars=self.BAOpars), k, mu, z
        )

        if cfg.settings["verbosity"] > 1:
            tps = time()
            print("Shot-noise obtained in {} seconds".format(tps - trsd))

        Pk_ref = rsd * Pk + Ps
        uP = Pk_ref.unit
        logPk_ref = np.log(Pk_ref.value)
        logk = np.log(k.to(u.Mpc**-1).value)

        #kompute using survey k's
        k = self.k
        outputshape = (*k.shape, *mu.shape, *z.shape)
        Pk_Obs = np.empty(outputshape)

        # Apply AP effect
        alpha_par = np.atleast_1d(self.alpha_parallel(z, self.BAOpars))
        alpha_perp = np.atleast_1d(self.alpha_perpendicular(z, self.BAOpars))
        kparr_ap = k[:, None, None] * mu[None, :, None] * alpha_par[None, None, :]
        kperp_ap = (
            k[:, None, None]
            * np.sqrt(1 - np.power(mu[None, :, None], 2))
            * alpha_perp[None, None, :]
        )
        # Compute related quantities
        k_ap = np.sqrt(np.power(kparr_ap, 2) + np.power(kperp_ap, 2))
        logk_ap = np.log(k_ap.to(u.Mpc**-1).value)
        mu_ap = kparr_ap / k_ap

        for iz, zi in enumerate(z):
            logk_ap_zi = logk_ap[:, :, iz].flatten()
            if (
                cfg.settings["QNLpowerspectrum"]
                or cfg.settings["do_RSD"]
                or self.halomodel.haloparams["v_of_M"]
            ):
                mu_ap_zi = mu_ap[:, :, iz].flatten()
                Pk_Obs[:, :, iz] = np.exp(
                    np.reshape(
                        bilinear_interpolate(
                            logk, mu, logPk_ref[:, :, iz], logk_ap_zi, mu_ap_zi
                        ),
                        (*k.shape, *mu.shape),
                    )
                )
            else:
                Pk_Obs[:, :, iz] = np.exp(
                    np.reshape(
                        linear_interpolate(logk, logPk_ref[:, 0, iz], logk_ap_zi),
                        (*k.shape, *mu.shape),
                    )
                )
        Pk_Obs = Pk_Obs * uP

        Fnu = np.ones(outputshape)
        if cfg.settings["Smooth_resolution"]:
            # The dampning from resolution is to be computed without any cosmolgy dependance
            F_parr = np.reshape(self.survey_specs.F_parr(k, mu, z, self.nuObs), outputshape)
            F_perp = np.reshape(self.survey_specs.F_perp(k, mu, z), outputshape)
            Fnu = F_parr * F_perp

        self.Pk_Obs = (
            (alpha_par * np.power(alpha_perp, 2))[None, None, :] * Pk_Obs * Fnu
        )

        if cfg.settings["verbosity"] > 1:
            tap = time()
            print(
                "Alcock-Paczynski projection performed in {} seconds".format(tap - tps)
            )

        if cfg.settings["Smooth_window"]:
            self.Pk_true = copy(self.Pk_Obs)
            self.Pk_Obs = self.convolved_Pk()

        if cfg.settings["verbosity"] > 1:
            print(
                "Observed power spectrum obtained in {} seconds".format(time() - tstart)
            )

    def compute_power_spectra_moments(self):
        """
        This function computes the power spectrum monopole, quadropole and hexadecapole
        For other moments it creats a new callable function to compute them
        """
        # grid quantities
        mu = self.mu
        Pobs = self.Pk_Obs

        def Pk_ell_moments(ell):
            norm = (2 * ell + 1) / 2
            L_ell = legendre(ell)
            return (
                simpson(y=Pobs * norm * L_ell(mu)[None, :, None], x=mu, axis=1)
                * Pobs.unit
            )

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
        for iz, zi in enumerate(z):
            interp_per_z.append(RectBivariateSpline(logk, mu, logP[:, :, iz]))

        def Pk_from_grid(pk, pmu):
            pk = np.atleast_1d(pk)
            pmu = np.atleast_1d(pmu)
            logpk = np.log(pk.to(uk).value)
            fromgrid = np.array([interp(logpk, pmu) for interp in interp_per_z])
            result = np.exp(fromgrid) * uP
            return np.squeeze(result)

        return Pk_from_grid


##############
# Numba Part #
##############


@njit(
    "(float64[::1], float64[::1], "
    + "float64[::1], float64[::1], float64[::1], "
    + "float64[:,:], float64[:,:])",
    parallel=True,
)
def convolve(k, mu, q, muq, deltaphi, P, W):
    # Check input sizes
    kl, mul = P.shape
    assert kl == k.size, "k should be the same size as axis 0 of P"
    assert mul == mu.size, "mu should be the same size as axis 1 of P"
    ql, muql = W.shape
    deltaphil = deltaphi.size
    assert ql == q.size, "q should be the same size as axis 0 of W"
    assert muql == muq.size, "muq should be the same size as axis 1 of W"

    # create Return array to be filled in parallel
    Pconv = np.empty_like(P, dtype=np.float64)
    for ik in prange(kl):
        for imu in prange(mul):
            # use q, muq, deltaphi and obtain abs k-q and the polar angle of k-q
            abskminusq = np.empty((ql, muql, deltaphil))
            mukminusq = np.empty((ql, muql, deltaphil))
            for iq in range(ql):
                for imuq in range(muql):
                    for ideltaphi in range(deltaphil):
                        kmq, mukmq, _ = addVectors(
                            k[ik], mu[imu], deltaphi[ideltaphi], q[iq], -mu[imuq], np.pi
                        )
                        abskminusq[iq, imuq, ideltaphi] = kmq
                        mukminusq[iq, imuq, ideltaphi] = mukmq

            # flatten the axis last axis first
            abskminusq = abskminusq.flatten()
            mukminusq = mukminusq.flatten()
            # interpolate the logP on mu logk and fill with new values
            logPkminusq = bilinear_interpolate(
                np.log(k), mu, np.log(P), np.log(abskminusq), mukminusq
            )
            logPkminusq = np.reshape(logPkminusq, (ql, muql, deltaphil))
            # Do the 3D trapezoid integration
            q_integrand = np.empty(ql)
            for iq in range(ql):
                muq_integrand = np.empty(muql)
                for imuq in range(muql):
                    phi_integrand = (
                        1
                        / (2 * np.pi) ** 3
                        * q[iq] ** 2
                        * (np.abs(W[iq, imuq]) ** 2)
                        * np.exp(logPkminusq[iq, imuq, :])
                    )
                    muq_integrand[imuq] = np.trapz(phi_integrand, deltaphi)
                q_integrand[iq] = np.trapz(muq_integrand, muq)
            Pconv[ik, imu] = np.trapz(q_integrand * q, np.log(q))
    return Pconv
