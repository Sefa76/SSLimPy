from copy import copy

import numpy as np
from astropy import units as u
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import minimize_scalar

from SSLimPy.cosmology.astro import AstroFunctions
from SSLimPy.LIMsurvey.cgf_inversion import CGFInverter
from SSLimPy.LIMsurvey.tracer_pdf import TracerPDF
from SSLimPy.utils.utils import *


class MatterDensityPDF:
    """
    Matter density PDF in a spherical top-hat cell, from LDT:
      - linear variance of the spherically-smoothed density field
      - spherical collapse mapping (contraction principle)
      - rate function Psi(delta), with the explicit non-linear rescaling
      - CGF via Legendre transform
      - PDF via saddle point (fast, standard) or exact FFT inversion
        (SSLimPy.LIMsurvey.cgf_inversion.CGFInverter)

    Parameters
    ----------
    astro : AstroFunctions
        Provides the cosmology and halo model.
    pdf_params : dict, optional
        Numerical settings, see _set_defaults() for the full list and
        defaults.
    """

    def __init__(self, astro: AstroFunctions, pdf_params: dict = dict()):
        self.cfg = astro.cfg

        self.cosmology = astro.cosmology
        self.halomodel = astro.halomodel
        self.astro = astro

        self.tracer = self.halomodel.tracer

        self.pdf_params = copy(pdf_params)
        self._set_defaults()

    def _set_defaults(self):
        # Spherical collapse mapping. nu=21/13 (Bernardeau 1994)
        # (Bernardeau & Reimberg 2016; Uhlemann et al. 2016).
        self.pdf_params.setdefault("nu_collapse", 21.0 / 13.0)

        # delta grid used to build the rate function / CGF. `None` means
        # "auto-scale from sigma_linear(R,z)" (see _auto_delta_range()) --
        # the natural width of the PDF can range from << 1 to >> 1
        # depending on R, z, so a single fixed absolute default is not
        # robust; set explicit numbers here only if you want to override
        # the auto-scaling.
        self.pdf_params.setdefault("n_delta", 500)
        self.pdf_params.setdefault("delta_min", None)
        self.pdf_params.setdefault("delta_max", None)

        # number of "sigmas" (sqrt(sigma2_linear(R,z))) the auto-scaled
        # delta grid extends to on the overdense side; the underdense side
        # is capped at delta_min_floor regardless (see _auto_delta_range()).
        self.pdf_params.setdefault("n_sigma_delta_range", 20.0)
        # Hard floor for the underdense side of the auto-scaled delta grid.
        # -0.9 (not -1) by default: Psi(delta) genuinely diverges as
        # delta -> -1 (tau(rho) -> -infinity faster than sigma_L^2(R_ini)
        # grows, for realistic LCDM P(k)); pushing this floor too close to
        # -1 blows up the dynamic range of Psi enough to degrade
        # CGFInverter's polynomial fit. Raise it back towards -1 only if you
        # specifically need the extreme-void tail resolved, and check
        # CGFInverter.fit_err / RankWarning if you do.
        self.pdf_params.setdefault("delta_min_floor", -0.98)

        # lambda grid used for the Legendre-transform CGF (cgf/build_cgf_grid).
        # `None` -> auto-scale from Psi'(delta) at the edges of delta_grid
        # (see build_cgf_grid()).
        self.pdf_params.setdefault("n_lambda", 500)
        self.pdf_params.setdefault("lambda_min", None)
        self.pdf_params.setdefault("lambda_max", None)

        # exact (FFT) inversion settings, see cgf_inversion.CGFInverter
        # ("effective mapping" method). ell_max_fft=None -> auto-scale as
        # n_sigma_ell_range / sigma_linear.
        #
        # Resolution note: the returned delta-grid spacing is
        # d_delta = pi/ell_max, INDEPENDENT of n_fft (n_fft only sets the
        # total delta-range covered, not how finely it is sampled) -- so
        # d_delta/sigma = pi/n_sigma_ell_range controls how smooth the
        # returned P(delta) looks. 100 gives ~30 points per sigma. Raising
        # this further improves resolution but pushes the complex
        # continuation to larger |ell| -- watch for CGFInverter's
        # extrapolation warning and back off (or pass an explicit, smaller
        # ell_max) if it fires.
        self.pdf_params.setdefault("order_tau", 20)
        self.pdf_params.setdefault("n_fft", 2**15)
        self.pdf_params.setdefault("ell_max_fft", None)
        self.pdf_params.setdefault("n_sigma_ell_range", 100.0)

        # Automatic self-consistency checks in pdf_exact(validate=True, the
        # default) -- see its docstring. These are heuristic thresholds,
        # not proofs of correctness; tighten or loosen them if you find
        # they fire too often/rarely for your use case.
        self.pdf_params.setdefault("validate_max_residual_tol", 1e-5)
        self.pdf_params.setdefault("validate_norm_tol", 0.02)
        self.pdf_params.setdefault("validate_peak_window_n_sigma", 0.5)
        self.pdf_params.setdefault("validate_peak_rel_err_tol", 0.05)

    # ============================================================
    # Variance
    # ============================================================

    def sigma2_linear(self, R, z, tracer=None):
        """
        Linear variance of the matter density contrast in a real-space
        spherical top-hat of comoving radius R, at redshift z:

            sigma_L^2(R,z)

        Returns a dimensionless float (or array, matching R/z's shape).
        """
        if tracer is None:
            tracer = self.tracer
        sigma = self.halomodel.sigmaR_of_z(R, z, tracer=tracer)
        return sigma**2

    def sigma2_nonlinear(self, R, z, k=None, tracer=None):
        """
        Non-linear variance of the matter density contrast in a real-space
        spherical top-hat, using the non-linear (halofit-type) power
        spectrum:

            sigma_NL^2(R,z) = int (k^2 dk)/(2 pi^2) P_NL(k,z) W_3D^2(kR)

        Used by nonlinear_rescaling_factor() (Eq. 8 of Boyle et al. 2021).
        """
        if tracer is None:
            tracer = self.tracer
        if k is None:
            k = self.cosmology.k
        kval = np.atleast_1d(k.to(u.Mpc**-1).value)

        Pk = self.cosmology.matpow(
            kval * u.Mpc**-1, z, nonlinear=True, tracer=tracer
        )
        Pk_val = np.atleast_1d(Pk.to(u.Mpc**3).value)

        x = kval * R.to(u.Mpc).value
        W = smooth_W(x)

        integrand = kval**2 * Pk_val * W**2 / (2.0 * np.pi**2)
        return float(simpson(integrand, x=kval))

    # ============================================================
    # Spherical collapse mapping
    # ============================================================

    def rho_from_tau_spherical(self, tau, nu=None):
        """
        Spherical collapse mapping (Bernardeau 1994 fitting form):

            rho = (1 - tau/nu)^(-nu),   delta = rho - 1
        """
        if nu is None:
            nu = self.pdf_params["nu_collapse"]
        return (1.0 - tau / nu) ** (-nu)

    def tau_from_rho_spherical(self, rho, nu=None):
        """
        Inverse spherical collapse mapping:

            tau(rho) = nu * [1 - rho^(-1/nu)]
        """
        if nu is None:
            nu = self.pdf_params["nu_collapse"]
        return nu * (1.0 - rho ** (-1.0 / nu))

    def R_ini_from_R_spherical(self, R, rho):
        """
        Mass conservation for a sphere:

            R_ini = R * rho^(1/3)
        """
        return R * rho ** (1.0 / 3.0)

    # ============================================================
    # Rate function
    # ============================================================

    def rate_function(self, delta, R, z, nu=None):
        """
        Linear-order rate function (spherical analogue of Eq. 6 of Boyle
        et al. 2021):

            Psi_l(delta) = tau(rho)^2 / (2 sigma_L^2(R_ini))

        with rho = 1+delta and R_ini = R*rho^(1/3) (spherical mass
        conservation). Uses ONLY the linear variance -- see
        build_rate_function_grid(nonlinear=True) (the default) for the
        explicit non-linear rescaling applied on top of this.

        `delta` must be a scalar (see build_rate_function_grid for the
        vectorized grid version).
        """
        rho = 1.0 + delta
        tau = self.tau_from_rho_spherical(rho, nu=nu)
        R_ini = self.R_ini_from_R_spherical(R, rho)
        sigma2 = self.sigma2_linear(R_ini, z)
        return tau**2 / (2.0 * sigma2)

    def nonlinear_rescaling_factor(self, R, z, k=None):
        """
        c = sigma_L^2(R,z) / sigma_NL^2(R,z), evaluated at the FIXED final
        (R,z) -- the explicit non-linear rescaling of Boyle et al. (2021)
        Eq. 8, applied as a single multiplicative correction to the whole
        rate function:

            Psi_nl(delta) = c * Psi_l(delta)

        (the Legendre-dual of Eq. 8's rescaling of the CGF itself,
        phi_nl(y) = c * phi_l(y/c) -- the two are equivalent, but rescaling
        Psi directly, with no argument change needed, is simpler to
        implement). Applied by default in build_rate_function_grid()
        (nonlinear=True).
        """
        sigma2_l = self.sigma2_linear(R, z)
        sigma2_nl = self.sigma2_nonlinear(R, z, k=k)
        return float(sigma2_l / sigma2_nl)

    def _auto_delta_range(self, R, z):
        """
        Estimate a sensible (delta_min, delta_max) for the rate-function
        grid from sigma_linear(R,z), used whenever pdf_params["delta_min"
        "/"delta_max"] are left at their default (None). The LINEAR
        variance is always used here purely to set the grid extent,
        regardless of whether the caller wants the nonlinear rate function.
        """
        sigma = float(np.sqrt(self.sigma2_linear(R, z)))
        n_sigma = self.pdf_params["n_sigma_delta_range"]
        delta_max = max(n_sigma * sigma, 3.0)
        delta_min = max(-n_sigma * sigma, self.pdf_params["delta_min_floor"])
        return delta_min, delta_max, sigma

    def build_rate_function_grid(self, R, z, nu=None, nonlinear=True):
        """Build the (delta, Psi(delta)) grid used by pdf_saddle()/
        pdf_exact() -- rate_function(), optionally rescaled to the
        non-linear variance via nonlinear_rescaling_factor() (Eq. 8,
        default True).

        The grid extent auto-scales from sigma_linear(R,z) unless
        pdf_params["delta_min"/"delta_max"] are set explicitly (see
        _auto_delta_range()).
        """
        delta_min = self.pdf_params["delta_min"]
        delta_max = self.pdf_params["delta_max"]
        if delta_min is None or delta_max is None:
            auto_min, auto_max, _ = self._auto_delta_range(R, z)
            delta_min = auto_min if delta_min is None else delta_min
            delta_max = auto_max if delta_max is None else delta_max

        delta_grid = np.linspace(
            delta_min, delta_max, self.pdf_params["n_delta"]
        )
        Psi_grid = np.array(
            [self.rate_function(d, R, z, nu=nu) for d in delta_grid]
        )
        if nonlinear:
            c = self.nonlinear_rescaling_factor(R, z)
            Psi_grid = c * Psi_grid
        return delta_grid, Psi_grid

    # ============================================================
    # CGF via Legendre transform
    # ============================================================

    def cgf(self, lam, delta_grid, Psi_grid):
        """
        phi(lambda) = sup_delta [lambda*delta - Psi(delta)]

        (Legendre-Fenchel transform of the rate function, evaluated by
        direct 1D extremization.

        Psi(delta) is not guaranteed to be convex everywhere -- this can
        make the objective lambda*delta-Psi(delta) genuinely multimodal in
        delta for some lambda, and a single bounded 1D optimizer (which
        assumes unimodality) can then lock onto the WRONG local optimum,
        giving a phi(lambda) with spurious discontinuous jumps as lambda
        varies. To guard against this, first coarsely grid-search
        the objective over delta_grid itself for a good starting bracket,
        then refine locally. pdf_saddle() does NOT go
        through this -- it uses Psi(delta), Psi''(delta) directly.
        """
        Psi_interp = interp1d(
            delta_grid,
            Psi_grid,
            kind="cubic",
            bounds_error=False,
            fill_value=np.inf,
        )

        def objective(delta):
            return -(lam * delta - Psi_interp(delta))

        obj_grid = -(lam * delta_grid - Psi_grid)
        i_best = int(np.argmin(obj_grid))
        lo = delta_grid[max(i_best - 2, 0)]
        hi = delta_grid[min(i_best + 2, len(delta_grid) - 1)]
        if lo == hi:
            return -obj_grid[i_best]

        result = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
        best_val = -result.fun
        if best_val < -obj_grid[i_best]:
            best_val = -obj_grid[i_best]
        return best_val

    def build_cgf_grid(self, delta_grid, Psi_grid, lambda_grid=None):
        """Evaluate phi(lambda) on a grid of lambda (see cgf()).

        If pdf_params["lambda_min"/"lambda_max"] are None (default), the
        range is auto-scaled from Psi'(delta) evaluated at the edges of
        delta_grid -- i.e. exactly the lambda range spanned by the already-
        built rate function grid.
        """
        if lambda_grid is None:
            lambda_min = self.pdf_params["lambda_min"]
            lambda_max = self.pdf_params["lambda_max"]
            if lambda_min is None or lambda_max is None:
                dspl = UnivariateSpline(
                    delta_grid, Psi_grid, s=0, k=4
                ).derivative(1)
                auto_max = float(dspl(delta_grid[-1]))
                auto_min = float(dspl(delta_grid[0]))
                lambda_max = auto_max if lambda_max is None else lambda_max
                lambda_min = auto_min if lambda_min is None else lambda_min
            lambda_grid = np.linspace(
                lambda_min, lambda_max, self.pdf_params["n_lambda"]
            )
        phi = np.array(
            [self.cgf(lam, delta_grid, Psi_grid) for lam in lambda_grid]
        )
        return lambda_grid, phi

    # ============================================================
    # PDF -- saddle point (leading order, always available)
    # ============================================================

    def pdf_saddle(self, R, z, nu=None, nonlinear=True, delta_eval=None):
        """
        Leading-order (in 1/sigma^2) saddle-point PDF, directly in delta:

            P(delta) = sqrt(Psi''(delta) / 2pi) * exp(-Psi(delta))

        with Psi = build_rate_function_grid()'s rate function (+ Eq. 8
        rescaling if nonlinear=True, the default). NaN wherever
        Psi''(delta) <= 0 -- a genuine large-deviations phenomenon (the
        saddle point becomes non-unique there), not a numerical error; see
        pdf_exact() for a method that remains well-defined there too.

        Returns
        -------
        delta_eval, pdf_eval : ndarray
            NOT renormalized; integrate to check numerical accuracy. May
            contain NaN where Psi is not locally convex (see note above).
        """
        delta_grid, Psi_grid = self.build_rate_function_grid(
            R, z, nu=nu, nonlinear=nonlinear
        )
        spl = UnivariateSpline(delta_grid, Psi_grid, s=0, k=4)
        ddspl = spl.derivative(2)

        if delta_eval is None:
            delta_eval = delta_grid
        delta_eval = np.atleast_1d(delta_eval)

        Psi_eval = spl(delta_eval)
        Psipp_eval = ddspl(delta_eval)

        pdf_eval = np.full_like(delta_eval, np.nan, dtype=float)
        valid = Psipp_eval > 0
        pdf_eval[valid] = np.sqrt(Psipp_eval[valid] / (2.0 * np.pi)) * np.exp(
            -Psi_eval[valid]
        )
        return delta_eval, pdf_eval

    # ============================================================
    # PDF -- exact (inverse Laplace transform of the CGF)
    # ============================================================

    def pdf_exact(
        self,
        R,
        z,
        nu=None,
        nonlinear=True,
        order=None,
        n_fft=None,
        ell_max=None,
        validate=True,
    ):
        """
        Exact PDF via the inverse Laplace transform of the CGF, using the
        "effective mapping" complex continuation of
        SSLimPy.LIMsurvey.cgf_inversion.CGFInverter (Bernardeau & Valageas
        2000, as spelled out in Appendix C of Boyle et al. 2021. Works
        directly with Psi(delta) from build_rate_function_grid() -- no
        variable reparametrisation needed.

        Automatic validation (validate=True, the default): after computing
        the PDF, three cheap self-consistency checks are run and reported
        via warnings.warn() if any of them fail -- so you get an
        automatic heads-up for a new (R,z) you have not manually checked,
        instead of having to remember to call cross_check() yourself every
        time:

          1. max_residual (the continuation's own self-consistency check)
             is not too large.
          2. The returned PDF integrates close to 1 (a generic sign that
             the delta/ell grids are well matched to this sigma).
          3. pdf_saddle() and pdf_exact() agree well in a NARROW window
             right around delta=0 (scaled to sigma).

        Returns
        -------
        delta_eval, pdf_eval : ndarray
            NOT renormalized; integrate to check numerical accuracy.
        max_residual : float
            Diagnostic from CGFInverter.pdf(); should be tiny (<<1e-6).
        inverter : CGFInverter
            Kept around so the caller can run inverter.cross_check(...)
            against pdf_saddle() in the bulk.
        """
        order = order or self.pdf_params["order_tau"]
        n_fft = n_fft or self.pdf_params["n_fft"]

        delta_grid, Psi_grid = self.build_rate_function_grid(
            R, z, nu=nu, nonlinear=nonlinear
        )

        ell_max = (
            ell_max if ell_max is not None else self.pdf_params["ell_max_fft"]
        )
        if ell_max is None:
            _, _, sigma = self._auto_delta_range(R, z)
            ell_max = self.pdf_params["n_sigma_ell_range"] / max(sigma, 1e-6)

        inverter = CGFInverter(delta_grid, Psi_grid, order=order)
        delta_eval, pdf_eval, max_residual = inverter.pdf(
            n=n_fft, ell_max=ell_max
        )

        if validate:
            self._validate_pdf_exact(
                R, z, delta_eval, pdf_eval, max_residual, inverter
            )

        return delta_eval, pdf_eval, max_residual, inverter

    def _validate_pdf_exact(
        self, R, z, delta_eval, pdf_eval, max_residual, inverter
    ):
        """Cheap automatic self-consistency checks for pdf_exact(), see its
        docstring. Issues warnings.warn() (never raises) so a bad check
        never silently blocks a result."""
        import warnings as _warnings

        tag = f"pdf_exact(R={R}, z={z})"

        max_res_tol = self.pdf_params["validate_max_residual_tol"]
        if max_residual > max_res_tol:
            _warnings.warn(
                f"{tag}: max_residual={max_residual:.2e} exceeds "
                f"pdf_params['validate_max_residual_tol']={max_res_tol:.1e} -- the "
                "complex continuation may not have converged cleanly. Try a "
                "wider delta_grid (n_sigma_delta_range) or a different order_tau.",
                stacklevel=3,
            )

        norm = float(np.trapezoid(pdf_eval, delta_eval))
        norm_tol = self.pdf_params["validate_norm_tol"]
        if abs(norm - 1.0) > norm_tol:
            _warnings.warn(
                f"{tag}: returned PDF integrates to {norm:.4f} (expected ~1, "
                f"tolerance {norm_tol:.1%}) -- the delta/ell grids may be poorly "
                "matched to this sigma. Check ell_max/n_fft, or the auto-scaling "
                "in _auto_delta_range().",
                stacklevel=3,
            )

        _, _, sigma = self._auto_delta_range(R, z)
        peak_width = self.pdf_params["validate_peak_window_n_sigma"] * sigma
        cc = inverter.cross_check(
            delta_eval, pdf_eval, delta_min=-peak_width, delta_max=peak_width
        )
        peak_tol = self.pdf_params["validate_peak_rel_err_tol"]
        if np.all(np.isnan(cc["rel_err"])):
            _warnings.warn(
                f"{tag}: could not compare against pdf_saddle right around the "
                f"peak (delta in [{-peak_width:.3g}, {peak_width:.3g}]) -- Psi is "
                "non-convex there too, which is unusual this close to delta=0. "
                "Inspect build_rate_function_grid()'s output directly.",
                stacklevel=3,
            )
        elif np.nanmax(cc["rel_err"]) > peak_tol:
            _warnings.warn(
                f"{tag}: pdf_exact disagrees with pdf_saddle by up to "
                f"{np.nanmax(cc['rel_err']):.1%} within +/-{peak_width:.3g} of the "
                f"peak (tolerance {peak_tol:.1%}) -- both methods should agree "
                "tightly this close to delta=0 regardless of sigma, so this is "
                "a genuine red flag, not expected tail behaviour. Try a higher "
                "order_tau, or inspect build_rate_function_grid()'s Psi(delta) "
                "directly for something unusual.",
                stacklevel=3,
            )


# ============================================================
# Single-tracer intensity/temperature PDF: P(I|1)
# ============================================================


def _luminosity_moments(astro, z, R, L_grid=None):
    """dn/dL(L,z) (astro.haloluminosityfunction -- unified across
    model_type="LF"/"ML"), integrated to give the total number density
    n_bar and the mean tracer count N_bar in a spherical voxel of radius R.

    This is the ONE place N_bar is computed from the luminosity function --
    both build_P_I_given_1() and VoxelIntensity._get_tracer_pdf() call this
    (rather than each integrating dn/dL separately), so the N_bar behind
    P(N_t) (TracerPDF) and the N_bar/normalisation behind P(I|1) can never
    silently drift apart.
    """
    if L_grid is None:
        L_grid = astro.L

    dndL = astro.haloluminosityfunction(
        L_grid, np.atleast_1d(z)
    )  # Mpc^-3 Lsun^-1
    dndL = np.squeeze(dndL)

    n_bar = np.trapezoid(dndL, L_grid)  # Mpc^-3
    if n_bar <= 0:
        raise ValueError(
            "n_bar <= 0: check astro.L's range against your luminosity "
            "function's support (e.g. its Lmin cutoff) -- integrating dn/dL "
            "gave zero or negative density."
        )

    V = 4.0 / 3.0 * np.pi * R**3
    N_bar = (n_bar * V).to(u.dimensionless_unscaled).value
    return L_grid, dndL, n_bar, N_bar


def build_P_I_given_1(astro, z, R, L_grid=None):
    """Build P(I|1) -- the intensity/temperature PDF of a SINGLE tracer --
    from an AstroFunctions instance, reusing its own haloluminosityfunction
    (dn/dL) and CLT (the luminosity-to-temperature conversion).

    CLT(z) converts a LUMINOSITY DENSITY (Lsun/Mpc^3, as in Lavg/Tavg) to a
    mean brightness temperature -- it is NOT a per-source L-to-T conversion
    by itself -- to get the intensity contribution of ONE source with 
    luminosity L sitting in a voxel of physical volume V, you must additionally
    divide by V (spreading that one source's luminosity over the voxel, 
    the same logic as Tavg spreading the ensemble luminosity density over an 
    implicit unit volume):

        Delta_T_single(L) = CLT(z) * L / V

    Parameters
    ----------
    astro : SSLimPy.cosmology.astro.AstroFunctions
    z : float or Quantity
        Redshift.
    R : Quantity (length)
        Voxel radius -- MUST be the same R used for MatterDensityPDF/
        TracerPDF (see VoxelIntensity, which enforces this automatically).
    L_grid : Quantity (luminosity), optional
        Defaults to astro.L. See _luminosity_moments()'s docstring on the
        shared N_bar computation, and the Lmin-cutoff note below.

        Cutoff note: astro.L's own Lmin (class default 10 Lsun) is well
        below typical luminosity-function cutoffs (e.g. "SchCut"'s default
        Lmin=5000 Lsun) -- so the physical cutoff is applied by the
        luminosity-function MODEL itself (returning ~0 below its Lmin), not
        by truncating the grid. If you changed either default, check this
        is still true -- plot dn/dL near your cutoff and confirm astro.L
        resolves it.

    Returns
    -------
    I_grid : Quantity (uK)
        The single-source intensity grid (log-spaced, matching astro.L --
        NOT directly FFT-ready; VoxelIntensity.intensity_pdf() resamples
        this onto a uniform linear grid before convolving).
    P_I_given_1 : Quantity (1/uK)
        Normalised PDF over I_grid (integrates to 1).
    N_bar : float
        Mean tracer count in the voxel -- see _luminosity_moments().
    """
    L_grid, dndL, n_bar, N_bar = _luminosity_moments(
        astro, z, R, L_grid=L_grid
    )

    P_L_given_1 = dndL / n_bar  # Lsun^-1, integrates to 1 over L_grid

    V = 4.0 / 3.0 * np.pi * R**3
    CLT_val = astro.CLT(z)
    I_grid = (CLT_val * L_grid / V).to(u.uK)

    # Jacobian for L -> I (CLT_val/V is a constant scale factor here, not
    # L-dependent): P(I) dI = P(L) dL => P(I) = P(L) * dL/dI = P(L)*V/CLT_val
    P_I_given_1 = (P_L_given_1 * V / CLT_val).to(1.0 / u.uK)

    return I_grid, P_I_given_1, N_bar


# ============================================================
# VoxelIntensity: orchestration layer
# ============================================================


class VoxelIntensity:
    """
    Orchestration layer for the voxel intensity distribution (VID):
    combines MatterDensityPDF (P(delta_m)), TracerPDF (P(N_t|delta_m),
    P(N_t)), and the single-source intensity PDF (P(I|1), above) into the
    full P(I).

    Method (Scheuer 1957; Condon 1974 "P(D)" source-confusion analysis;
    Breysse et al. 2017's VID, generalised here to the non-Poisson P(N_t)
    this project's bias+stochasticity model gives, rather than assuming a
    bare Poisson count):

        P(I)   = sum_N P(I|N) P(N_t=N)
        P(I|N) = IFT[ FT(P(I|1))^N ]

    N_bar (both for TracerPDF's P(N_t) and for normalising P(I|1)) comes
    from the SAME luminosity-function integral in both cases.

    Parameters
    ----------
    astro : AstroFunctions
    vid_params : dict, optional
        "pdf_params": forwarded to MatterDensityPDF.
        "bias_params": forwarded to TracerPDF (b1_G, b2_G, alpha0/1/2,
            bias_model, bias_mode -- everything TracerPDF accepts except
            matter_pdf/N_bar, which VoxelIntensity supplies itself).
    """

    def __init__(self, astro: AstroFunctions, vid_params: dict = dict()):
        self.astro = astro
        self.vid_params = copy(vid_params)
        self.matter_pdf = MatterDensityPDF(
            astro, pdf_params=self.vid_params.get("pdf_params", dict())
        )
        self.bias_params = self.vid_params.get("bias_params", dict())
        # TracerPDF bakes N_bar in at construction time, and N_bar depends
        # on R (via the luminosity-function integral) -- so, unlike
        # matter_pdf (built once, R/z passed per call), a TracerPDF
        # instance is built (and cached) per (R,z) here.
        self._tracer_pdf_cache = {}

    def _get_tracer_pdf(self, R, z):
        key = (repr(R), float(z))
        cached = self._tracer_pdf_cache.get(key)
        if cached is not None:
            return cached
        _, _, _, N_bar = _luminosity_moments(self.astro, z, R)
        tracer = TracerPDF(self.matter_pdf, N_bar=N_bar, **self.bias_params)
        self._tracer_pdf_cache[key] = tracer
        return tracer

    def emitter_number_pdf(self, R, z, N_max=None, **kwargs):
        """P(N_t): the marginal number-of-emitters PDF (TracerPDF.marginal_pmf,
        the CosMomentum-validated discrete route -- see that method's
        docstring). For the CONDITIONAL P(N_t|delta_m) instead, use
        self._get_tracer_pdf(R, z).pmf_Ne_given_delta_m_integer(...) or
        .pdf_Ne_given_delta_m(...) directly.
        """
        tracer = self._get_tracer_pdf(R, z)
        return tracer.marginal_pmf(R, z, N_max=N_max, **kwargs)

    def intensity_pdf(
        self,
        R,
        z,
        N_max=None,
        n_fft_intensity=2**15,
        n_sigma_intensity=15.0,
        tail_prob=1e-6,
        validate=True,
        validate_rel_err_tol=0.05,
    ):
        """
        The full voxel intensity distribution P(I).

        Parameters
        ----------
        N_max : int, optional
            Forwarded to TracerPDF.marginal_pmf().
        n_fft_intensity : int
            Number of points in the (uniform, linear) I grid used for the
            FFT convolution. astro.L (and hence P(I|1)'s native grid) is
            LOG-spaced -- FFT convolution needs uniform spacing, so P(I|1)
            is resampled onto a fresh linear grid of this size before use.
        n_sigma_intensity, tail_prob : float
            Together set the width of that linear I grid, as the LARGER of
            two independent estimates:
              (a) an aggregate, CLT-motivated width from the compound
                  distribution's own analytic moments (law of total
                  variance for a random sum: Var(I) = E[N]*Var(I|1) +
                  Var(N)*E[I|1]^2): I_max_a = E[I] + n_sigma_intensity*std(I)
              (b) an empirical percentile of P(I|1) ITSELF (from its native,
                  log-spaced grid, which resolves a steep luminosity
                  function's tail far better than a variance-based estimate
                  can): the I value below which fits a fraction
                  1 - tail_prob/N_max_actual of P(I|1)'s own probability --
                  i.e. chosen so that, even drawing up to N_max_actual iid
                  single-source values, the chance ANY ONE of them exceeds
                  this is only ~tail_prob.
        validate : bool
            If True (default), after computing P(I), checks the returned
            distribution's own mean against the analytic E[I] = E[N]*E[I|1]
            (which holds regardless of whether N is Poisson) and warns if
            they disagree by more than validate_rel_err_tol.

        Returns
        -------
        I_grid : Quantity (uK), uniform linear grid, starting at 0.
        P_I : Quantity (1/uK)
        """
        tracer = self._get_tracer_pdf(R, z)
        N_grid, P_N = tracer.marginal_pmf(R, z, N_max=N_max)
        norm_N = np.sum(P_N)
        mean_N = np.sum(N_grid * P_N) / norm_N
        var_N = np.sum((N_grid - mean_N) ** 2 * P_N) / norm_N
        N_max_actual = int(N_grid.max())

        I1_grid, P_I1_native, _N_bar = build_P_I_given_1(self.astro, z, R)
        I1_val = I1_grid.to(u.uK).value
        P_I1_val = P_I1_native.to(1.0 / u.uK).value
        mean_I1 = np.trapezoid(I1_val * P_I1_val, I1_val)
        var_I1 = np.trapezoid((I1_val - mean_I1) ** 2 * P_I1_val, I1_val)

        # (a) aggregate, CLT-motivated width (law of total variance):
        mean_I_expected = mean_N * mean_I1
        var_I_expected = mean_N * var_I1 + var_N * mean_I1**2
        I_max_aggregate = mean_I_expected + n_sigma_intensity * np.sqrt(
            max(var_I_expected, 0.0)
        )

        # (b) empirical single-source tail percentile, from the NATIVE
        # (log-spaced) P(I|1) grid, which resolves a steep luminosity
        # function's tail far better than any variance-based estimate:
        cdf = np.concatenate(
            [
                [0.0],
                np.cumsum(
                    0.5 * (P_I1_val[1:] + P_I1_val[:-1]) * np.diff(I1_val)
                ),
            ]
        )
        cdf = cdf / cdf[-1]
        target = min(1.0 - tail_prob / max(N_max_actual, 1), 1.0 - 1e-12)
        I_max_percentile = float(np.interp(target, cdf, I1_val))

        I_max = max(I_max_aggregate, I_max_percentile)
        if I_max <= 0.0:
            I_max = I1_val.max()  # degenerate fallback (e.g. mean_N == 0)

        I_uniform = np.linspace(0.0, I_max, n_fft_intensity)
        dI = I_uniform[1] - I_uniform[0]

        P_I1_uniform = np.interp(
            I_uniform, I1_val, P_I1_val, left=0.0, right=0.0
        )
        norm = np.trapezoid(P_I1_uniform, I_uniform)
        if norm <= 0:
            raise ValueError(
                "P(I|1) resampled onto the uniform FFT grid integrates to "
                "<= 0 -- n_fft_intensity is likely too small to resolve the "
                "single-source distribution at all. Raise n_fft_intensity."
            )
        P_I1_uniform /= norm

        phi1 = np.fft.fft(P_I1_uniform) * dI

        # Fourier-space power-sum: phi(w) = sum_N P(N) * phi1(w)^N, built
        # via a running product (phi1^N at step N) rather than N_max
        # separate inverse FFTs, or an (N_max, n_fft) array of powers --
        # this needs only O(n_fft_intensity) memory regardless of N_max
        # (same lesson as TracerPDF.marginal_pmf's own chunking, applied
        # here in a form that never needed chunking in the first place).
        phi = np.zeros_like(phi1)
        current_power = np.ones_like(phi1)  # phi1^0
        for N, pN in zip(N_grid.astype(int), P_N / norm_N):
            if N > 0:
                current_power = current_power * phi1
            phi += pN * current_power

        P_I = np.real(np.fft.ifft(phi)) / dI

        if validate:
            mean_I_fft = np.sum(I_uniform * P_I) * dI
            if mean_I_expected > 0:
                rel_err = abs(mean_I_fft - mean_I_expected) / mean_I_expected
                if rel_err > validate_rel_err_tol:
                    import warnings as _warnings

                    _warnings.warn(
                        f"intensity_pdf(R={R}, z={z}): E[I] from the computed "
                        f"P(I) ({mean_I_fft:.4g}) disagrees with the analytic "
                        f"E[N]*E[I|1] ({mean_I_expected:.4g}) by {rel_err:.1%} "
                        f"(tolerance {validate_rel_err_tol:.1%}) -- the I grid "
                        "likely under-resolves P(I|1) or truncates its tail. "
                        "Try raising n_fft_intensity, n_sigma_intensity, or "
                        "lowering tail_prob.",
                        stacklevel=2,
                    )

        return I_uniform * u.uK, P_I / u.uK
