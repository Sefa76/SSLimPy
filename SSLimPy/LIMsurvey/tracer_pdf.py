import warnings

import numpy as np
from scipy.integrate import simpson
from scipy.signal import medfilt
from scipy.special import gammaln


class TracerPDF:
    """
    Conditional and marginal PDF of tracer (emitter) counts, given a
    MatterDensityPDF instance for the underlying matter density in the
    same spherical cell.

    Parameters
    ----------
    matter_pdf : SSLimPy.LIMsurvey.voxel_intensity.MatterDensityPDF
        Provides the matter density PDF/rate function machinery this
        class builds on (delta_L(delta_m) via tau_from_rho_spherical,
        pdf_saddle/pdf_exact for the matter PDF itself, and the
        CGFInverter pdf_exact() returns).
    N_bar : float
        Mean number of tracers (emitters) in the cell, i.e. tracer number
        density times cell volume (4/3 pi R^3 for the spherical cell
        MatterDensityPDF assumes). You supply this -- SSLimPy's
        AstroFunctions/luminosity-function machinery is the natural place
        to compute it, not this module.
    b1_G, b2_G : float
        Bias parameters, in whichever convention bias_model expects
        (Lagrangian for "gaussian_lagrangian"/"additive_lagrangian",
        Eulerian for "eulerian_quadratic") -- fit once per tracer
        sample/redshift, not per R.
    alpha0, alpha1, alpha2 : float
        Quadratic shot-noise (stochasticity) parameters,
        alpha(delta_m) = alpha0 + alpha1*delta_m + alpha2*delta_m^2
        alpha0=1, alpha1=alpha2=0 recovers plain Poisson sampling of the
        biased field.
    bias_model : {"gaussian_lagrangian", "eulerian_quadratic", "additive_lagrangian"}
        Default "gaussian_lagrangian".
    bias_mode : {"exact", "tree"}
        Only used when bias_model="additive_lagrangian"
        Default "exact".
    nu : float, optional
        Spherical-collapse parameter passed through to
        tau_from_rho_spherical (defaults to matter_pdf's own default).
    """

    def __init__(
        self,
        matter_pdf,
        N_bar,
        b1_G,
        b2_G,
        alpha0=1.0,
        alpha1=0.0,
        alpha2=0.0,
        bias_model="gaussian_lagrangian",
        bias_mode="exact",
        nu=None,
    ):
        if bias_model not in (
            "gaussian_lagrangian",
            "eulerian_quadratic",
            "additive_lagrangian",
        ):
            raise ValueError(
                "bias_model must be 'gaussian_lagrangian', 'eulerian_quadratic', "
                "or 'additive_lagrangian'"
            )
        if bias_mode not in ("exact", "tree"):
            raise ValueError('bias_mode must be "exact" or "tree"')
        self.matter_pdf = matter_pdf
        self.N_bar = float(N_bar)
        self.b1_G = float(b1_G)
        self.b2_G = float(b2_G)
        self.alpha0 = float(alpha0)
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.bias_model = bias_model
        self.bias_mode = bias_mode
        self.nu = nu
        # Warn about 1+delta_e < 0 clipping ONCE per
        # instance (GalaxySample::error_flag_negative_density), not once
        # per point -- only relevant for bias_model="additive_lagrangian".
        self._additive_clip_warned = False
        # Cache for _get_exact_bias_grid(): the exact contour-integral
        # bias-term grids are expensive (a full CGFInverter.pdf()-sized
        # FFT, twice) -- keyed by (R, z, nu, nonlinear, order, n_fft, ell_max).
        self._exact_bias_cache = {}

    # ============================================================
    # Shared building block: delta_L(delta_m)
    # ============================================================

    def _delta_L(self, delta_m, nu=None):
        """delta_L(delta_m): the inverse spherical collapse mapping,
        reusing MatterDensityPDF.tau_from_rho_spherical directly (same
        object as the "tau" in rate_function() -- delta_L is exactly the
        linear-theory density that collapses, under spherical dynamics,
        to the given nonlinear delta_m)."""
        rho = 1.0 + np.atleast_1d(delta_m)
        return self.matter_pdf.tau_from_rho_spherical(
            rho, nu=nu if nu is not None else self.nu
        )

    def _matter_pdf_grid(
        self, R, z, nu=None, nonlinear=True, method="saddle", **pdf_kwargs
    ):
        """Fetch a (delta_grid, pdf_grid) pair from the underlying
        MatterDensityPDF, dropping any NaN (non-convex-region) points.

        method="exact" (pdf_exact, the full FFT-based reconstruction) vs.
        "saddle" (pdf_saddle, the tree-level approximation). Prefer
        "exact" whenever the tail matters (it usually does for a marginal
        P(N_e)); "saddle" is faster and fine for e.g. a single conditional
        P(N_e|delta_m) query near the peak.
        """
        if method == "saddle":
            delta_grid, pdf_grid = self.matter_pdf.pdf_saddle(
                R,
                z,
                nu=nu if nu is not None else self.nu,
                nonlinear=nonlinear,
                **pdf_kwargs,
            )
        elif method == "exact":
            delta_grid, pdf_grid, _, _ = self.matter_pdf.pdf_exact(
                R,
                z,
                nu=nu if nu is not None else self.nu,
                nonlinear=nonlinear,
                **pdf_kwargs,
            )
        else:
            raise ValueError("method must be 'saddle' or 'exact'")
        valid = np.isfinite(pdf_grid)
        return delta_grid[valid], pdf_grid[valid]

    def _sigma_m2(self, R, z, nonlinear=True):
        """The matter variance entering the Gaussian-Lagrangian/Eulerian
        bias kernels' normalisation (the non-linear variance by default,
        matching how the underlying MatterDensityPDF rate function is
        itself normally used)."""
        if nonlinear:
            return self.matter_pdf.sigma2_nonlinear(R, z)
        return self.matter_pdf.sigma2_linear(R, z)

    @staticmethod
    def _mask_nonfinite(values, delta_m_grid, tag):
        """Replace any non-finite entries of `values` (evaluated at the
        corresponding points of `delta_m_grid`) with 0, warning once with
        where/how-many. This is the failure mode of division-normalised
        bias models right at the delta_m -> -1 edge (overflow -> inf) --
        expected physics at the boundary (MG25 themselves exclude the most
        extreme low-density quantiles from their own analysis for the same
        reason), not a bug; the point of this guard is only to make that
        explicit and loud rather than an unexplained NaN."""
        bad = ~np.isfinite(values)
        if np.any(bad):
            bad_dm = np.atleast_1d(delta_m_grid)
            if bad_dm.shape == bad.shape:
                bad_dm = bad_dm[bad]
                lo, hi = float(np.min(bad_dm)), float(np.max(bad_dm))
                where = f"delta_m in [{lo:.4g}, {hi:.4g}]"
            else:
                where = "delta_m near the -1 edge"
            warnings.warn(
                f"{tag}: {int(np.sum(bad))} of {values.size} point(s) were "
                f"non-finite ({where}) and have been masked to zero before "
                "integrating. Expected physics at the delta_m -> -1 edge, "
                "not a bug -- MG25 exclude the most extreme low-density "
                "quantiles from their own analysis for the same reason.",
                stacklevel=3,
            )
        return np.where(bad, 0.0, values)

    # ============================================================
    # Bias model 1: gaussian_lagrangian (default, recommended)
    # ============================================================

    def _f_L_kernel(self, delta_m, sigma_m2, nu=None):
        """(1+delta_m) * f_L(delta_L(delta_m)), UNNORMALISED (before
        dividing by its own mean, see bias_function()) -- MG25 Eq. 21-22,
        the renormalised Gaussian Lagrangian bias of Stucker et al.
        (2024)."""
        delta_m = np.atleast_1d(delta_m)
        delta_L = self._delta_L(delta_m, nu=nu)
        b1, b2 = self.b1_G, self.b2_G
        prefac = np.exp(-(b1**2) / (2.0 * b2)) / np.sqrt(1.0 + b2 * sigma_m2)
        expo = b2 * (b1 / b2 + delta_L) ** 2 / (2.0 * (1.0 + b2 * sigma_m2))
        return (1.0 + delta_m) * prefac * np.exp(expo)

    def _bias_gaussian_lagrangian(
        self,
        delta_m,
        R,
        z,
        nu=None,
        nonlinear=True,
        method="saddle",
        **pdf_kwargs,
    ):
        # This model's own normalisation integral ALWAYS uses "saddle",
        # regardless of what `method` the caller passed (e.g. marginal_pmf()
        # defaulting to "exact" for ITS OWN outer delta_m grid).
        sigma_m2 = self._sigma_m2(R, z, nonlinear=nonlinear)
        delta_grid, pdf_grid = self._matter_pdf_grid(
            R, z, nu=nu, nonlinear=nonlinear, method="saddle", **pdf_kwargs
        )
        norm = simpson(pdf_grid, x=delta_grid)
        kernel_grid = self._f_L_kernel(delta_grid, sigma_m2, nu=nu)
        kernel_grid = self._mask_nonfinite(
            kernel_grid,
            delta_grid,
            tag=f"bias_function(R={R}, z={z}) normalisation integral",
        )
        mean_kernel = simpson(kernel_grid * pdf_grid, x=delta_grid) / norm

        kernel_at_delta_m = self._f_L_kernel(delta_m, sigma_m2, nu=nu)
        result = kernel_at_delta_m / mean_kernel
        if np.any(~np.isfinite(result)):
            warnings.warn(
                f"bias_function(R={R}, z={z}): {int(np.sum(~np.isfinite(result)))} "
                f"of {result.size} queried delta_m value(s) give a non-finite bias "
                "-- these query points are right at/beyond the delta_m -> -1 "
                "numerical edge.",
                stacklevel=3,
            )
        return result

    # ============================================================
    # Bias model 2: eulerian_quadratic
    # ============================================================

    def bias_function_eulerian(self, delta_m, R, z, nonlinear=True):
        """1 + <delta_e|delta_m> via the simpler quadratic Eulerian bias
        model (VU26 Eq. 22 / MG25 Eq. 17):
            <delta_e|delta_m> = b1*delta_m + (b2/2)*(delta_m^2 - sigma_m^2)
        """
        delta_m = np.atleast_1d(delta_m)
        sigma_m2 = self._sigma_m2(R, z, nonlinear=nonlinear)
        delta_e = self.b1_G * delta_m + 0.5 * self.b2_G * (
            delta_m**2 - sigma_m2
        )
        return 1.0 + delta_e

    # ============================================================
    # Bias model 3: additive_lagrangian (bias_mode="tree" or "exact")
    # ============================================================

    def _bias_terms_tree(self, delta_m, nu=None):
        """bias_term_1(delta_m) = (1+delta_m)*delta_L(delta_m)
        bias_term_2(delta_m) = bias_term_1(delta_m)*delta_L(delta_m)
        (FlatInhomogeneousUniverseLCDM.cpp lines ~1931-1932), evaluated
        pointwise at the tree-level saddle point."""
        delta_m = np.atleast_1d(delta_m).astype(float)
        dL = self._delta_L(delta_m, nu=nu)
        bt1 = (1.0 + delta_m) * dL
        bt2 = bt1 * dL
        return bt1, bt2

    @staticmethod
    def _reject_local_outliers(values, window=15, factor=5.0):
        """Replace isolated single-point spikes in `values` (already in
        delta_grid order, as returned by CGFInverter.conditional_expectation())
        with NaN, via a local median filter: a point further than `factor`
        times the local median absolute deviation from the local median is
        treated as noise, not signal. Found necessary because such points
        can still pass conditional_expectation()'s own floor_rel screen
        (they're finite, just numerically marginal)."""
        values = np.asarray(values, dtype=float)
        if values.size < window:
            return values
        finite = np.isfinite(values)
        if not np.any(finite):
            return values
        filled = np.where(finite, values, 0.0)
        k = window if window % 2 == 1 else window + 1
        med = medfilt(filled, kernel_size=k)
        mad = medfilt(np.abs(filled - med), kernel_size=k)
        scale = np.maximum(mad, 1e-8 * np.maximum(np.abs(med), 1.0))
        outlier = finite & (np.abs(values - med) > factor * scale)
        cleaned = values.copy()
        cleaned[outlier] = np.nan
        return cleaned

    def _get_exact_bias_grid(
        self,
        R,
        z,
        nu=None,
        nonlinear=True,
        order=None,
        n_fft=None,
        ell_max=None,
    ):
        """Build (once, cached) <bias_term_1|delta_m>, <bias_term_2|delta_m>
        via the EXACT contour-integral machinery of
        CGFInverter.conditional_expectation() -- reusing the SAME
        CGFInverter instance matter_pdf.pdf_exact() already builds for the
        plain matter PDF at this (R,z); no separate/duplicate CGF machinery.

        Cached because this is expensive: two conditional_expectation()
        calls, each a full CGFInverter.pdf()-sized FFT.
        """
        order = order or self.matter_pdf.pdf_params["order_tau"]
        n_fft = n_fft or self.matter_pdf.pdf_params["n_fft"]
        ell_max_resolved = (
            ell_max
            if ell_max is not None
            else self.matter_pdf.pdf_params["ell_max_fft"]
        )
        if ell_max_resolved is None:
            _, _, sigma = self.matter_pdf._auto_delta_range(R, z)
            ell_max_resolved = self.matter_pdf.pdf_params[
                "n_sigma_ell_range"
            ] / max(sigma, 1e-6)

        cache_key = (repr(R), z, nu, nonlinear, order, n_fft, ell_max_resolved)
        if cache_key in self._exact_bias_cache:
            return self._exact_bias_cache[cache_key]

        _, _, _, inverter = self.matter_pdf.pdf_exact(
            R,
            z,
            nu=nu if nu is not None else self.nu,
            nonlinear=nonlinear,
            order=order,
            n_fft=n_fft,
            ell_max=ell_max_resolved,
            validate=False,
        )

        delta_m_tau_full = inverter.delta_grid_sorted
        tau_full = inverter.tau_grid
        tau_min, tau_max = tau_full.min(), tau_full.max()
        margin_tau = 0.03 * (tau_max - tau_min)
        safe = (tau_full > tau_min + margin_tau) & (
            tau_full < tau_max - margin_tau
        )
        if np.sum(safe) < 10:
            safe = np.ones_like(tau_full, dtype=bool)
        delta_m_tau = delta_m_tau_full[safe]
        tau_sub = tau_full[safe]

        dL_tau = self._delta_L(delta_m_tau, nu=nu)
        bt1_tau = (1.0 + delta_m_tau) * dL_tau
        bt2_tau = bt1_tau * dL_tau

        inverter.fit_observable("bias_term_1", bt1_tau, tau=tau_sub)
        inverter.fit_observable("bias_term_2", bt2_tau, tau=tau_sub)

        delta_grid, cond_b1, _r1 = inverter.conditional_expectation(
            "bias_term_1", n=n_fft, ell_max=ell_max_resolved
        )
        _, cond_b2, _r2 = inverter.conditional_expectation(
            "bias_term_2", n=n_fft, ell_max=ell_max_resolved
        )

        cond_b1 = self._reject_local_outliers(cond_b1)
        cond_b2 = self._reject_local_outliers(cond_b2)

        result = (delta_grid, cond_b1, cond_b2)
        self._exact_bias_cache[cache_key] = result
        return result

    def _bias_terms_exact(
        self,
        delta_m,
        R,
        z,
        nu=None,
        nonlinear=True,
        order=None,
        n_fft=None,
        ell_max=None,
    ):
        """<bias_term_1|delta_m>, <bias_term_2|delta_m> -- the exact
        conditional expectation where _get_exact_bias_grid() can reliably
        compute it, falling back to _bias_terms_tree() (the raw point
        evaluation) elsewhere."""
        delta_m = np.atleast_1d(delta_m).astype(float)
        delta_grid, cond_b1, cond_b2 = self._get_exact_bias_grid(
            R,
            z,
            nu=nu,
            nonlinear=nonlinear,
            order=order,
            n_fft=n_fft,
            ell_max=ell_max,
        )
        good = np.isfinite(cond_b1) & np.isfinite(cond_b2)
        delta_grid_g, cond_b1_g, cond_b2_g = (
            delta_grid[good],
            cond_b1[good],
            cond_b2[good],
        )

        bt1 = np.empty_like(delta_m)
        bt2 = np.empty_like(delta_m)
        if delta_grid_g.size == 0:
            reliable = np.zeros_like(delta_m, dtype=bool)
        else:
            lo, hi = delta_grid_g.min(), delta_grid_g.max()
            reliable = (delta_m >= lo) & (delta_m <= hi)

        if np.any(reliable):
            bt1[reliable] = np.interp(
                delta_m[reliable], delta_grid_g, cond_b1_g
            )
            bt2[reliable] = np.interp(
                delta_m[reliable], delta_grid_g, cond_b2_g
            )
        if np.any(~reliable):
            bt1_fb, bt2_fb = self._bias_terms_tree(delta_m[~reliable], nu=nu)
            bt1[~reliable] = bt1_fb
            bt2[~reliable] = bt2_fb
        return bt1, bt2

    def bias_function_additive_lagrangian(
        self, delta_m, R, z, nu=None, nonlinear=True
    ):
        """1 + <delta_e|delta_m> via the classic additive Lagrangian bias
        expansion (Friedrich et al. 2022:

            <delta_e|delta_m> = delta_m + b1_G*bias_term_1(delta_m)
                                + (b2_G/2)*bias_term_2(delta_m)

        bias_term_1/2 come from _bias_terms_tree() or _bias_terms_exact()
        depending on self.bias_mode -- see module docstring.

        Not guaranteed positive: 1+delta_e can go below 0 for extreme
        delta_m. Handled by CLIPPING to 0 and
        warning ONCE per instance (CosMomentum-style, matching their own
        error_flag_negative_density).
        """
        delta_m = np.atleast_1d(delta_m).astype(float)
        if self.bias_mode == "tree":
            bt1, bt2 = self._bias_terms_tree(delta_m, nu=nu)
        else:
            bt1, bt2 = self._bias_terms_exact(
                delta_m, R, z, nu=nu, nonlinear=nonlinear
            )

        delta_e = delta_m + self.b1_G * bt1 + 0.5 * self.b2_G * bt2
        bias = 1.0 + delta_e

        bad = bias < 0.0
        if np.any(bad):
            if not self._additive_clip_warned:
                self._additive_clip_warned = True
                warnings.warn(
                    f"bias_function(R={R}, z={z}, bias_model='additive_lagrangian', "
                    f"bias_mode='{self.bias_mode}'): {int(np.sum(bad))} of {bias.size} "
                    "point(s) gave 1+delta_e < 0 and were clipped to 0 (delta_e "
                    "clipped to -1), exactly as in CosMomentum's "
                    "GalaxySample::return_P_of_N_given_delta_g. Shown only once "
                    "per instance.",
                    stacklevel=3,
                )
            bias = np.where(bad, 0.0, bias)
        return bias

    # ============================================================
    # Bias dispatcher
    # ============================================================

    def bias_function(
        self,
        delta_m,
        R,
        z,
        nu=None,
        nonlinear=True,
        method="saddle",
        **pdf_kwargs,
    ):
        """1 + <delta_e|delta_m>: dispatches to whichever bias_model this
        instance was constructed with.

        Returns
        -------
        bias : ndarray, same shape as delta_m
        """
        if self.bias_model == "eulerian_quadratic":
            return self.bias_function_eulerian(
                delta_m, R, z, nonlinear=nonlinear
            )
        if self.bias_model == "additive_lagrangian":
            return self.bias_function_additive_lagrangian(
                delta_m, R, z, nu=nu, nonlinear=nonlinear
            )
        return self._bias_gaussian_lagrangian(
            delta_m,
            R,
            z,
            nu=nu,
            nonlinear=nonlinear,
            method=method,
            **pdf_kwargs,
        )

    def delta_g(
        self,
        delta_m,
        R,
        z,
        nu=None,
        nonlinear=True,
        method="saddle",
        **pdf_kwargs,
    ):
        """delta_g(delta_m) = bias_function(...) - 1. Used internally by
        N_max() and marginal_pmf()."""
        return (
            self.bias_function(
                delta_m,
                R,
                z,
                nu=nu,
                nonlinear=nonlinear,
                method=method,
                **pdf_kwargs,
            )
            - 1.0
        )

    # ============================================================
    # Stochasticity model (conditional variance)
    # ============================================================

    def shot_noise(self, delta_m):
        """alpha(delta_m) = alpha0 + alpha1*delta_m + alpha2*delta_m^2
        (MG25 Eq. 24) -- the ratio of the conditional variance to the
        conditional mean of the tracer count; 1 for Poisson sampling."""
        delta_m = np.atleast_1d(delta_m)
        return self.alpha0 + self.alpha1 * delta_m + self.alpha2 * delta_m**2

    # ============================================================
    # Conditional mean tracer count
    # ============================================================

    def mean_Ne_given_delta_m(
        self, delta_m, R, z, nu=None, nonlinear=True, method="saddle"
    ):
        """N_bar_e(delta_m) = N_bar * (1 + <delta_e|delta_m>)."""
        return self.N_bar * self.bias_function(
            delta_m, R, z, nu=nu, nonlinear=nonlinear, method=method
        )

    # ============================================================
    # Conditional PDF: P(N_e | delta_m) -- continuous density
    # ============================================================

    def log_pdf_Ne_given_delta_m(
        self, N_e, delta_m, R, z, nu=None, nonlinear=True, method="saddle"
    ):
        """log P(N_e | delta_m) (MG25 Eq. 14) -- the alpha-rescaled Poisson
        distribution, evaluated in log-space for numerical stability.

        N_e may be non-integer (a continuous generalisation of the Poisson
        PMF via the Gamma function in place of the factorial).
        """
        N_e = np.atleast_1d(N_e).astype(float)
        alpha = self.shot_noise(delta_m)
        N_bar_e = self.mean_Ne_given_delta_m(
            delta_m, R, z, nu=nu, nonlinear=nonlinear, method=method
        )
        x = N_e / alpha
        mu = N_bar_e / alpha
        # x*log(mu) at x==0 (N_e==0): the correct value is 0 (standard
        # 0*log(0)=0 convention), not numpy's 0*(-inf)=nan. mu==0 is the
        # ordinary result of bias_model="additive_lagrangian" clipping
        # 1+delta_e to 0 -- getting this right (rather than routing it
        # through a non-finite guard) is what keeps both pmf_Ne_given_delta_m_integer()
        # and marginal_pmf() correctly normalised in that regime (their row-
        # sum/renormalisation naturally recovers a clean point mass at N_e=0).
        with np.errstate(divide="ignore", invalid="ignore"):
            log_mu = np.log(mu)
            x_log_mu = x * log_mu
        zero_x = x == 0
        if np.any(zero_x):
            x_log_mu = np.where(
                np.broadcast_to(zero_x, x_log_mu.shape), 0.0, x_log_mu
            )
        return x_log_mu - mu - gammaln(x + 1.0) - np.log(alpha)

    @staticmethod
    def _poisson_pdf_core(N, mu, alpha):
        """Same alpha-rescaled Poisson formula as log_pdf_Ne_given_delta_m(),
        but taking PRECOMPUTED mu (=N_bar_e/alpha) and alpha directly instead
        of calling bias_function()/mean_Ne_given_delta_m() itself. Used by
        marginal_pmf() to compute the bias ONCE (vectorised over the whole
        delta_m grid) instead of once per delta_m row."""
        N = np.atleast_1d(N).astype(float)
        mu = np.atleast_1d(mu).astype(float)
        alpha = np.atleast_1d(alpha).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = N / alpha
            log_mu = np.log(mu)
            x_log_mu = x * log_mu
        zero_x = x == 0
        if np.any(zero_x):
            x_log_mu = np.where(
                np.broadcast_to(zero_x, x_log_mu.shape), 0.0, x_log_mu
            )
        return np.exp(x_log_mu - mu - gammaln(x + 1.0) - np.log(alpha))

    def pdf_Ne_given_delta_m(
        self, N_e, delta_m, R, z, nu=None, nonlinear=True, method="saddle"
    ):
        """P(N_e | delta_m) (MG25 Eq. 14-15) -- see log_pdf_Ne_given_delta_m().

        Returns
        -------
        pdf : ndarray
            NOT renormalized over N_e (analytically normalized by
            construction over the full continuous range; on a finite grid
            check np.trapezoid/np.sum as usual).
        """
        return np.exp(
            self.log_pdf_Ne_given_delta_m(
                N_e, delta_m, R, z, nu=nu, nonlinear=nonlinear, method=method
            )
        )

    def pmf_Ne_given_delta_m_integer(
        self,
        N_e_int,
        delta_m,
        R,
        z,
        nu=None,
        nonlinear=True,
        method="saddle",
        n_sub=21,
    ):
        """Proper discrete probability MASS at integer tracer counts
        N_e_int, by bin-integrating pdf_Ne_given_delta_m() over
        [N_e_int-0.5, N_e_int+0.5] with Simpson's rule.

        Parameters
        ----------
        N_e_int : array_like of int (or int-valued float), >= 0.
        n_sub : int
            Sub-points for the bin integral (Simpson's rule).

        Returns
        -------
        pmf : ndarray, same shape as N_e_int
        """
        N_e_int = np.atleast_1d(N_e_int).astype(float)
        if np.any(N_e_int < 0) or np.any(N_e_int != np.round(N_e_int)):
            raise ValueError("N_e_int must contain non-negative integers")

        delta_m_arr = np.atleast_1d(delta_m).astype(float)
        if len(N_e_int) > 1 and len(delta_m_arr) > 1:
            raise ValueError(
                "pmf_Ne_given_delta_m_integer: at most one of N_e_int, delta_m may have length > 1"
            )

        if len(delta_m_arr) > 1:
            pmf = np.empty_like(delta_m_arr)
            Ni = N_e_int[0]
            mu = self.mean_Ne_given_delta_m(
                delta_m_arr, R, z, nu=nu, nonlinear=nonlinear, method=method
            ) / self.shot_noise(delta_m_arr)
            zero_mu = mu <= 0.0

            lo = max(Ni - 0.5, 0.0)
            hi = Ni + 0.5
            sub = np.linspace(lo, hi, n_sub)
            dens = self.pdf_Ne_given_delta_m(
                sub[:, None],
                delta_m_arr[None, :],
                R,
                z,
                nu=nu,
                nonlinear=nonlinear,
                method=method,
            )
            bad_cols = ~np.all(np.isfinite(dens), axis=0) & ~zero_mu
            if np.any(bad_cols):
                dens = self._mask_nonfinite(
                    dens,
                    np.broadcast_to(delta_m_arr[None, :], dens.shape),
                    tag=f"pmf_Ne_given_delta_m_integer(R={R}, z={z}, N_e_int={Ni:g})",
                )
            pmf[:] = simpson(dens, x=sub, axis=0)
            if np.any(zero_mu):
                pmf[zero_mu] = 1.0 if Ni == 0 else 0.0
            return pmf

        pmf = np.empty_like(N_e_int)
        alpha_scalar = self.shot_noise(delta_m)
        mu_scalar = (
            self.mean_Ne_given_delta_m(
                delta_m, R, z, nu=nu, nonlinear=nonlinear, method=method
            )
            / alpha_scalar
        )
        if np.all(mu_scalar <= 0.0):
            pmf[:] = np.where(N_e_int == 0, 1.0, 0.0)
            return pmf
        # Reuse the ALREADY-COMPUTED mu_scalar/alpha_scalar for every bin
        # below via _poisson_pdf_core() -- calling pdf_Ne_given_delta_m()
        # again inside this loop would silently re-trigger bias_function()
        # (and, for bias_model="gaussian_lagrangian", its normalisation
        # integral) on EVERY one of potentially hundreds of N_e_int values,
        # even though delta_m is fixed for the whole loop.
        for i, Ni in enumerate(N_e_int):
            lo = max(Ni - 0.5, 0.0)
            hi = Ni + 0.5
            sub = np.linspace(lo, hi, n_sub)
            dens = self._poisson_pdf_core(sub, mu_scalar, alpha_scalar)
            pmf[i] = simpson(dens, x=sub)
        return pmf

    # ============================================================
    # Marginal P(N_e): CosMomentum route (literal discrete algorithm)
    # ============================================================

    @staticmethod
    def _variance_and_norm(delta_grid, pdf_grid):
        """Trapezoidal <delta_m^2> and normalisation of pdf_grid over
        delta_grid."""
        d_delta = np.diff(delta_grid)
        variance = np.sum(
            0.5
            * (
                delta_grid[:-1] ** 2 * pdf_grid[:-1]
                + delta_grid[1:] ** 2 * pdf_grid[1:]
            )
            * d_delta
        )
        norm = np.sum(0.5 * (pdf_grid[:-1] + pdf_grid[1:]) * d_delta)
        return float(variance), float(norm)

    def N_max(
        self,
        R,
        z,
        delta_m_grid=None,
        pdf_grid=None,
        method="exact",
        nu=None,
        nonlinear=True,
        N_max_ceiling=None,
    ):
        """Computational bound on N (5-sigma of both density and
        shot noise, assuming delta_m is lognormal with delta_0=-1), not a
        physical statement.

        N_max_ceiling : int, optional
        """
        if delta_m_grid is None:
            delta_m_grid, pdf_grid = self._matter_pdf_grid(
                R, z, nu=nu, nonlinear=nonlinear, method=method
            )
        variance, _norm = self._variance_and_norm(delta_m_grid, pdf_grid)

        var_Gauss = np.log(1.0 + variance)
        delta_max = np.exp(-0.5 * var_Gauss + 5.0 * np.sqrt(var_Gauss)) - 1.0

        if self.bias_model == "additive_lagrangian":
            b_lin = 1.0 + self.b1_G
            b_quad = self.b2_G + 8.0 / 21.0 * self.b1_G
            delta_g_max = b_lin * delta_max + 0.5 * b_quad * (
                delta_max**2 - variance
            )
            if b_quad < 0.0:
                delta_g_max = min(
                    -0.5 * b_lin / (0.5 * b_quad), b_lin * delta_max
                )
        elif self.bias_model == "eulerian_quadratic":
            b_lin, b_quad = self.b1_G, self.b2_G
            delta_g_max = b_lin * delta_max + 0.5 * b_quad * (
                delta_max**2 - variance
            )
            if b_quad < 0.0:
                delta_g_max = min(
                    -0.5 * b_lin / (0.5 * b_quad), b_lin * delta_max
                )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bias_at_max = self.bias_function(
                    np.array([delta_max]), R, z, nu=nu, nonlinear=nonlinear
                ).item()
            delta_g_max = bias_at_max - 1.0

        N_bar_max = self.N_bar * (1.0 + delta_g_max)
        galaxy_per_Poisson_halo = self.alpha0 + delta_max * self.alpha1
        if self.alpha2 > 0.0:
            galaxy_per_Poisson_halo += delta_max**2 * self.alpha2

        N_max_val = int(0.5 + min(N_bar_max, 1e15))
        if galaxy_per_Poisson_halo > 0.0:
            N_max_val += int(
                5.0
                * np.sqrt(
                    max(min(N_bar_max, 1e15), 0.0)
                    * min(galaxy_per_Poisson_halo, 1e15)
                )
            )
        N_max_val = max(N_max_val, 1)

        if N_max_ceiling is None:
            N_max_ceiling = max(10_000, int(2000 * (1.0 + self.N_bar)))
        if N_max_val > N_max_ceiling:
            warnings.warn(
                f"N_max(R={R}, z={z}, bias_model='{self.bias_model}'): computed "
                f"N_max={N_max_val:.4g} exceeds the safety ceiling {N_max_ceiling} "
                f"and was capped there (variance behind this estimate: {variance:.4g}, "
                f"delta_max: {delta_max:.4g}). This formula is exponentially "
                "sensitive to variance -- see this method's docstring. The usual "
                'fix is a narrower delta_m_grid (e.g. method="saddle"\'s) feeding '
                "the variance estimate, not raising N_max_ceiling.",
                stacklevel=2,
            )
            N_max_val = N_max_ceiling
        return N_max_val

    def _marginal_pmf_auto(
        self,
        R,
        z,
        delta_m_grid,
        pdf_grid,
        method="exact",
        nu=None,
        nonlinear=True,
        N_max_start=1000,
        rtol=1e-3,
        max_doublings=4,
    ):
        """Adaptive-N_max helper for marginal_pmf(N_max="auto").
        delta_m_grid/pdf_grid are
        fetched ONCE by the caller and reused for every doubling here, so
        the (possibly expensive, e.g. method="exact"'s CGFInverter) matter
        PDF is never recomputed mid-search -- only the cheaper per-N_max
        work in marginal_pmf() itself repeats.
        """
        N_max_try = max(int(N_max_start), 1)
        N_grid, P_of_N = self.marginal_pmf(
            R,
            z,
            N_max=N_max_try,
            delta_m_grid=delta_m_grid,
            method=method,
            nu=nu,
            nonlinear=nonlinear,
        )
        prev_sum = float(np.sum(P_of_N))
        for _ in range(max_doublings):
            N_max_try *= 2
            N_grid, P_of_N = self.marginal_pmf(
                R,
                z,
                N_max=N_max_try,
                delta_m_grid=delta_m_grid,
                method=method,
                nu=nu,
                nonlinear=nonlinear,
            )
            new_sum = float(np.sum(P_of_N))
            rel_change = abs(new_sum - prev_sum) / max(abs(prev_sum), 1e-300)
            if rel_change < rtol:
                return N_grid, P_of_N
            prev_sum = new_sum
        warnings.warn(
            f"marginal_pmf(R={R}, z={z}, N_max='auto'): did not converge to "
            f"rtol={rtol:.1e} within {max_doublings} doublings (final N_max="
            f"{N_max_try}, last relative change={rel_change:.2e}) -- returning "
            "the last (largest) attempt anyway, but it may still be under-"
            "resolved in N. Consider raising auto_max_doublings or "
            "auto_N_max_start, or check whether N_max() itself would be "
            "inflated for this (R,z) (see its own docstring) -- a genuinely "
            "non-converging search is often the same root cause.",
            stacklevel=2,
        )
        return N_grid, P_of_N

    def marginal_pmf(
        self,
        R,
        z,
        N_max=None,
        delta_m_grid=None,
        method="exact",
        nu=None,
        nonlinear=True,
        auto_N_max_start=1000,
        auto_rtol=1e-3,
        auto_max_doublings=4,
    ):
        """Marginal discrete PMF P(N_e):
          1. Evaluate P(N_e|delta_m) DIRECTLY at each integer N_e (reusing
             pdf_Ne_given_delta_m().
          2. Renormalise that row to sum to exactly 1.
          3. Integrate over delta_m with the TRAPEZOIDAL rule.

        Parameters
        ----------
        N_max : int, "auto", or None
            int: use this N_max directly.
            None (default): use self.N_max()'s formal 5-sigma estimate.
                Confirmed directly to often be MUCH more conservative than
                actually needed.
            "auto": adaptively DOUBLE N_max, starting from
                auto_N_max_start, until the marginal's own sum stops
                changing by more than auto_rtol between successive
                doublings.
        delta_m_grid : array_like, optional
            Defaults to matter_pdf's own grid.
        method : {"exact", "saddle"}
            P(delta_m) source -- "exact" strongly recommended..
        auto_N_max_start, auto_rtol, auto_max_doublings : only used when
            N_max="auto".

        Returns
        -------
        N_grid : ndarray, 0..N_max
        P_of_N : ndarray, same shape
        """
        dgrid, pgrid = self._matter_pdf_grid(
            R, z, nu=nu, nonlinear=nonlinear, method=method
        )
        if delta_m_grid is None:
            delta_m_grid, pdf_grid = dgrid, pgrid
        else:
            delta_m_grid = np.asarray(delta_m_grid, dtype=float)
            pdf_grid = np.interp(delta_m_grid, dgrid, pgrid)

        if N_max == "auto":
            return self._marginal_pmf_auto(
                R,
                z,
                delta_m_grid,
                pdf_grid,
                method=method,
                nu=nu,
                nonlinear=nonlinear,
                N_max_start=auto_N_max_start,
                rtol=auto_rtol,
                max_doublings=auto_max_doublings,
            )

        if N_max is None:
            N_max = self.N_max(
                R,
                z,
                delta_m_grid=delta_m_grid,
                pdf_grid=pdf_grid,
                nu=nu,
                nonlinear=nonlinear,
            )
        N_grid = np.arange(N_max + 1).astype(float)

        # Compute the bias ONCE, vectorised over the whole delta_m grid --
        # NOT inside the row loop below.
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )  # the one-time clip warning is expected
            N_bar_e_grid = self.mean_Ne_given_delta_m(
                delta_m_grid, R, z, nu=nu, nonlinear=nonlinear, method=method
            )
        alpha_grid = self.shot_noise(delta_m_grid)
        mu_grid = N_bar_e_grid / alpha_grid

        n_delta = len(delta_m_grid)
        n_N = N_max + 1

        # Process delta_m in CHUNKS, vectorised within each chunk -- not
        # one delta_m at a time and not the WHOLE (n_delta, n_N) array at
        # once.
        target_elements = 20_000_000  # ~160MB per float64 chunk array
        chunk_size = max(1, target_elements // max(n_N, 1))

        P_of_N = np.zeros(n_N)
        prev_row = prev_pdf = prev_dm = None
        for start in range(0, n_delta, chunk_size):
            end = min(start + chunk_size, n_delta)
            dm_chunk = delta_m_grid[start:end]
            mu_chunk = mu_grid[start:end]
            alpha_chunk = alpha_grid[start:end]
            pdf_chunk = pdf_grid[start:end]

            x = N_grid[None, :] / alpha_chunk[:, None]
            mu_b = mu_chunk[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                log_mu = np.log(mu_b)
                x_log_mu = x * log_mu
            x_log_mu = np.where(x == 0, 0.0, x_log_mu)
            rows = np.exp(
                x_log_mu
                - mu_b
                - gammaln(x + 1.0)
                - np.log(alpha_chunk[:, None])
            )
            rows = np.where(np.isfinite(rows), rows, 0.0)
            row_sums = rows.sum(axis=1, keepdims=True)
            rows = np.where(row_sums > 0.0, rows / row_sums, rows)

            # Trapezoidal integration needs consecutive PAIRS, including
            # the boundary between chunks -- carry the last row forward.
            if prev_row is not None:
                full_rows = np.vstack([prev_row[None, :], rows])
                full_pdf = np.concatenate([[prev_pdf], pdf_chunk])
                full_dm = np.concatenate([[prev_dm], dm_chunk])
            else:
                full_rows, full_pdf, full_dm = rows, pdf_chunk, dm_chunk

            weighted = full_rows * full_pdf[:, None]
            d_delta_chunk = np.diff(full_dm)
            if len(d_delta_chunk) > 0:
                P_of_N += np.sum(
                    0.5
                    * (weighted[:-1] + weighted[1:])
                    * d_delta_chunk[:, None],
                    axis=0,
                )

            prev_row, prev_pdf, prev_dm = rows[-1], pdf_chunk[-1], dm_chunk[-1]

        return N_grid, P_of_N
