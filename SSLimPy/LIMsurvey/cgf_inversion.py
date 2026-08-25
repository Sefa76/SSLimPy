import numpy as np
from numpy.polynomial import chebyshev as _chebyshev
from scipy.interpolate import UnivariateSpline as _UnivariateSpline


class CGFInverter:
    """Exact PDF from a tabulated rate function Psi(delta) (or a tabulated
    CGF phi(lambda), see from_cgf()), via inverse Laplace transform of the
    CGF (the "effective mapping" method of Bernardeau & Valageas 2000, as
    used in CosMomentum).

    Parameters
    ----------
    delta_grid : array_like
        Grid of density-contrast values on which the rate function is
        tabulated. Should be sorted and span comfortably beyond the region
        where the PDF is needed (including well into the delta<0 side,
        ideally close to delta=-1, since that is generally where the
        rate function is steepest and constrains the reliable range of the
        continuation).
    Psi_grid : array_like
        Rate function Psi(delta) evaluated on delta_grid. Must satisfy
        Psi >= 0, with Psi=0 only at the mean (delta=0 for a density
        contrast). Psi need NOT be convex everywhere.
    order : int, optional
        Degree of the Chebyshev polynomial fit to the effective mapping
        zeta(tau). Default 20 is normally ample (fit_err below machine
        precision in testing); raise it if `fit_err` is not small compared
        to the dynamic range of delta, lower it if you see a RankWarning
        from numpy (the fit is overdetermined for the smoothness of the
        input).

    Attributes
    ----------
    fit_err : float
        Maximum absolute deviation of the Chebyshev fit from the tabulated
        (tau, zeta) points. Inspect this; a large value means `order` needs
        adjusting or the input grid is under-resolved.
    """

    def __init__(self, delta_grid, Psi_grid, order=20):
        delta_grid = np.asarray(delta_grid, dtype=float)
        Psi_grid = np.asarray(Psi_grid, dtype=float)
        if delta_grid.shape != Psi_grid.shape:
            raise ValueError(
                "delta_grid and Psi_grid must have the same shape"
            )
        if np.any(Psi_grid < -1e-10):
            raise ValueError(
                "Psi_grid contains significantly negative values; a rate "
                "function must be >= 0 everywhere for this construction "
                "(tau = sign(lambda)*sqrt(2*Psi) requires Psi >= 0)."
            )

        # Keep the original tabulated grid so cross_check() can rebuild the
        # same real-axis Psi'' used for the saddle-point comparison.
        self._orig_delta_grid = delta_grid
        self._orig_Psi_grid = Psi_grid

        # lambda(delta) = Psi'(delta): use a smooth spline purely to get a
        # clean real-axis derivative (this spline is never evaluated at
        # complex arguments, so its conditioning near delta=-1 is not an
        # issue the way a complex-evaluated polynomial's would be).
        spl = _UnivariateSpline(delta_grid, Psi_grid, s=0, k=4)
        dspl = spl.derivative(1)

        lam = dspl(delta_grid)
        Psi = np.clip(Psi_grid, 0.0, None)
        tau = np.sign(lam) * np.sqrt(2.0 * Psi)
        zeta = delta_grid  # the effective mapping IS delta itself vs tau

        self._build_from_tau_zeta(tau, zeta, order)

    @classmethod
    def from_cgf(cls, lambda_grid, phi_grid, order=20):
        """
        Alternative constructor: build the effective-mapping machinery
        directly from a TABULATED CGF phi(lambda), rather than from a rate
        function Psi(delta) (the default constructor). Use this when
        phi(lambda) itself is your primary object.

        The effective mapping here is zeta(lambda) = phi'(lambda) (the
        saddle point delta*(lambda), from the standard Legendre duality),
        obtained via ONE numerical (spline) derivative of the tabulated
        phi(lambda); tau(lambda) then follows exactly as in the default
        constructor. Everything past that point is identical.

        Parameters
        ----------
        lambda_grid : array_like
            Grid of lambda values (real axis) on which phi is tabulated.
            Should be sorted and span comfortably beyond the region where
            the PDF is needed.
        phi_grid : array_like
            phi(lambda) evaluated on lambda_grid, with phi(0)=0.
        order : int, optional
            See __init__.

        Note: cross_check() is not available on an instance built this
        way (it needs a delta-space rate function to compare against,
        which this constructor does not have) -- it raises accordingly.
        """
        lambda_grid = np.asarray(lambda_grid, dtype=float)
        phi_grid = np.asarray(phi_grid, dtype=float)
        if lambda_grid.shape != phi_grid.shape:
            raise ValueError(
                "lambda_grid and phi_grid must have the same shape"
            )

        dspl_phi = _UnivariateSpline(
            lambda_grid, phi_grid, s=0, k=4
        ).derivative(1)
        zeta = dspl_phi(lambda_grid)
        combo = lambda_grid * zeta - phi_grid

        if np.any(combo < -1e-8 * max(np.max(np.abs(phi_grid)), 1.0)):
            raise ValueError(
                "lambda*phi'(lambda) - phi(lambda) is significantly negative "
                "somewhere; this should be >= 0 for a valid CGF (it equals "
                "Psi(delta*(lambda)) at the saddle point) -- check phi_grid, "
                "or that lambda_grid is finely/widely enough sampled for the "
                "spline derivative to be accurate."
            )

        tau = np.sign(lambda_grid) * np.sqrt(2.0 * np.clip(combo, 0.0, None))

        inst = cls.__new__(cls)
        inst._orig_delta_grid = None
        inst._orig_Psi_grid = None
        inst._build_from_tau_zeta(tau, zeta, order)
        return inst

    def _build_from_tau_zeta(self, tau, zeta, order):
        """Shared Chebyshev-fitting step for both constructors: fit the
        effective mapping zeta(tau), the single object everything else is
        derived from."""
        order_idx = np.argsort(tau)
        tau_s = tau[order_idx]
        zeta_s = zeta[order_idx]

        self._tau_grid = tau_s
        self._zeta_grid = zeta_s
        self._tau_bounds = (float(tau_s.min()), float(tau_s.max()))

        self.zeta_cheb = _chebyshev.Chebyshev.fit(
            tau_s, zeta_s, deg=order, domain=self._tau_bounds
        )
        self._dzeta_cheb = self.zeta_cheb.deriv(1)
        self._ddzeta_cheb = self.zeta_cheb.deriv(2)

        self.fit_err = float(np.max(np.abs(self.zeta_cheb(tau_s) - zeta_s)))

    # ------------------------------------------------------------------
    # Analytic relations derived from the single zeta(tau) fit
    # ------------------------------------------------------------------

    def _lambda_of_tau(self, tau):
        return tau / self._dzeta_cheb(tau)

    def _phi_of_tau(self, tau):
        return self._lambda_of_tau(tau) * self.zeta_cheb(tau) - 0.5 * tau**2

    # ------------------------------------------------------------------
    # Complex continuation: tau(ell) via coarse continuation + vectorized
    # Newton polish (the coarse pass is a Python loop and is the speed
    # bottleneck; keep n_coarse modest and let the polish step do the
    # precision work on the full, potentially much larger, FFT grid).
    # ------------------------------------------------------------------

    def _newton_tau_scalar(self, target_ell, tau0, maxit=100, tol=1e-11):
        tau = tau0
        for _ in range(maxit):
            zp = self._dzeta_cheb(tau)
            f = tau / zp - 1j * target_ell
            zpp = self._ddzeta_cheb(tau)
            fp = (zp - tau * zpp) / zp**2
            step = f / fp
            tau = tau - step
            if abs(step) < tol:
                return tau, True
        return tau, False

    def _coarse_tau_of_ell(self, ell_coarse):
        ell_coarse = np.asarray(ell_coarse, dtype=float)
        tau_out = np.zeros_like(ell_coarse, dtype=complex)
        for mask in (ell_coarse >= 0, ell_coarse <= 0):
            idx = np.where(mask)[0]
            idx = idx[np.argsort(np.abs(ell_coarse[idx]))]
            prev_tau = 0j
            for i in idx:
                t, ok = self._newton_tau_scalar(ell_coarse[i], prev_tau)
                tau_out[i] = t
                prev_tau = t
        return tau_out

    def _newton_tau_vectorized(self, ell, tau0, n_iter=8):
        tau = tau0.copy()
        for _ in range(n_iter):
            zp = self._dzeta_cheb(tau)
            f = tau / zp - 1j * ell
            zpp = self._ddzeta_cheb(tau)
            fp = (zp - tau * zpp) / zp**2
            tau = tau - f / fp
        return tau

    def phi_imag_axis(self, ell_grid, n_coarse=400, n_polish=8):
        """Evaluate phi(i*ell) for an array of real ell.

        Returns
        -------
        phi_vals : complex ndarray, same shape as ell_grid
        max_residual : float
            max_i |lambda(tau_i) - i*ell_i| over the returned grid -- a
            direct check that the continuation actually solved the target
            equation (does not by itself guarantee the fit represents the
            true CGF far into the tails; use cross_check() for that).
        """
        _, phi_vals, max_residual = self._tau_fine_and_phi(
            ell_grid, n_coarse=n_coarse, n_polish=n_polish
        )
        return phi_vals, max_residual

    def _tau_fine_and_phi(self, ell_grid, n_coarse=400, n_polish=8):
        """Shared continuation step behind phi_imag_axis(): also returns
        tau_fine (the complex tau at each ell), which conditional_expectation()
        needs to evaluate an observable(tau) fit at the SAME points, without
        repeating the (expensive) Newton continuation a second time."""
        ell_grid = np.asarray(ell_grid, dtype=float)
        ell_max = np.max(np.abs(ell_grid)) if ell_grid.size else 0.0

        ell_pos = np.linspace(0.0, ell_max, n_coarse // 2 + 1)
        ell_coarse = np.concatenate([-ell_pos[1:][::-1], ell_pos])
        tau_coarse = self._coarse_tau_of_ell(ell_coarse)

        re_interp = _UnivariateSpline(ell_coarse, tau_coarse.real, s=0, k=3)
        im_interp = _UnivariateSpline(ell_coarse, tau_coarse.imag, s=0, k=3)
        tau0 = re_interp(ell_grid) + 1j * im_interp(ell_grid)

        tau_fine = self._newton_tau_vectorized(ell_grid, tau0, n_iter=n_polish)
        residual = np.abs(self._lambda_of_tau(tau_fine) - 1j * ell_grid)

        # The Chebyshev fit is only validated reasonably close to the tau
        # domain it was built from (self._tau_bounds); warn if the
        # continuation strayed far beyond it (meaning ell_max is too large
        # for the delta_grid/lambda_grid this inverter was built from).
        tau_scale = max(
            abs(self._tau_bounds[0]), abs(self._tau_bounds[1]), 1e-300
        )
        max_excursion = max(np.max(np.abs(tau_fine)) - tau_scale, 0.0)
        if max_excursion > 0.5 * tau_scale:
            import warnings as _warnings

            _warnings.warn(
                f"CGFInverter: the continuation for ell_max={np.max(np.abs(ell_grid)):.3g} "
                f"required |tau| up to {np.max(np.abs(tau_fine)):.3g}, well beyond the "
                f"fitted tau domain {self._tau_bounds} (scale {tau_scale:.3g}) -- the "
                "Chebyshev fit is being extrapolated well outside where it was validated. "
                "Rebuild the inverter from a wider delta_grid/lambda_grid, or reduce "
                "ell_max, and re-check.",
                stacklevel=2,
            )

        return tau_fine, self._phi_of_tau(tau_fine), float(np.max(residual))

    @property
    def tau_grid(self):
        """This inverter's own real tau grid (sorted) -- build observable
        arrays on THIS grid (not on the original, possibly differently-
        ordered delta_grid/Psi_grid passed to __init__) before calling
        fit_observable()."""
        return self._tau_grid

    @property
    def delta_grid_sorted(self):
        """delta values (== zeta(tau_grid)) in the same order as tau_grid --
        the natural companion to tau_grid for building observable(delta)
        arrays to pass to fit_observable()."""
        return self._zeta_grid

    def fit_observable(self, name, values, tau=None, order=None):
        """Fit an auxiliary observable O as a Chebyshev polynomial in tau --
        the same kind of fit zeta(tau) itself gets, enabling O to be
        evaluated at the complex tau values conditional_expectation() needs.

        Parameters
        ----------
        name : str
            Key to retrieve this fit via conditional_expectation(name=...).
        values : array_like
            The observable, tabulated at the `tau` points below.
        tau : array_like, optional
            Where `values` is tabulated. Defaults to self.tau_grid (this
            inverter's own full grid) -- pass a NARROWER subset here (with
            matching `values`) when the observable is only well-behaved
            away from the domain edges.
        order : int, optional
            Chebyshev degree; defaults to the same order used for zeta(tau).

        Returns
        -------
        fit_err : float
            max|fitted - values| on the given tau points -- a sanity check,
            not a guarantee of good behaviour BETWEEN those points.
        """
        tau = self._tau_grid if tau is None else np.asarray(tau, dtype=float)
        values = np.asarray(values, dtype=float)
        if values.shape != tau.shape:
            raise ValueError(
                f"values and tau must have the same shape -- got "
                f"{values.shape} and {tau.shape}"
            )
        order = order if order is not None else self.zeta_cheb.degree()
        domain = (float(np.min(tau)), float(np.max(tau)))
        cheb = _chebyshev.Chebyshev.fit(tau, values, deg=order, domain=domain)
        if not hasattr(self, "_observable_chebs"):
            self._observable_chebs = {}
            self._observable_domains = {}
        self._observable_chebs[name] = cheb
        self._observable_domains[name] = domain
        fit_err = float(np.max(np.abs(cheb(tau) - values)))
        return fit_err

    def conditional_expectation(
        self,
        name,
        n=2**15,
        ell_max=200.0,
        n_coarse=400,
        n_polish=8,
        floor_rel=1e-3,
    ):
        """<O(delta')|delta> for the observable fitted under `name` via
        fit_observable(), on the SAME delta grid pdf() would return for
        these settings.

        Parameters
        ----------
        floor_rel : float, optional
            Points are masked to NaN wherever the plain PDF is below
            floor_rel * max(pdf_grid), OR negative.

        Returns
        -------
        delta_grid, cond_exp_grid, max_residual : as pdf(), but cond_exp_grid
            is NaN wherever the plain PDF is too small/negative to divide by
            safely (see floor_rel) -- <O|delta> is not meaningfully
            recoverable this way deep in the tails regardless; use
            pdf_saddle-level physical judgement (or a tree-level bias) in
            that regime.
        """
        if (
            not hasattr(self, "_observable_chebs")
            or name not in self._observable_chebs
        ):
            raise KeyError(
                f"no observable fitted under name={name!r} -- call "
                "fit_observable(name, values) first"
            )
        obs_cheb = self._observable_chebs[name]

        ell = (np.arange(n) - n // 2) * (2.0 * ell_max / n)
        tau_fine, phi_vals, max_residual = self._tau_fine_and_phi(
            ell, n_coarse=n_coarse, n_polish=n_polish
        )
        plain_integrand = np.exp(phi_vals)

        # Clip tau_fine's real part to the OBSERVABLE's own fitted domain
        # before evaluating it -- a Chebyshev polynomial extrapolates
        # explosively fast just past its fit domain
        obs_domain = getattr(self, "_observable_domains", {}).get(name)
        if obs_domain is not None:
            lo, hi = obs_domain
            tau_for_obs = np.clip(tau_fine.real, lo, hi) + 1j * tau_fine.imag
        else:
            tau_for_obs = tau_fine
        weighted_integrand = obs_cheb(tau_for_obs) * plain_integrand

        dell = ell[1] - ell[0]

        def _fft(integrand):
            F = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(integrand)) * dell)
            return np.real(F) / (2.0 * np.pi)

        delta_grid = np.fft.fftshift(np.fft.fftfreq(n, d=dell)) * 2.0 * np.pi
        pdf_grid = _fft(plain_integrand)
        weighted_grid = _fft(weighted_integrand)

        floor = floor_rel * np.max(pdf_grid)
        cond_exp = np.full_like(pdf_grid, np.nan)
        ok = (
            pdf_grid > floor
        )  # excludes both "too small" AND negative pdf_grid
        cond_exp[ok] = weighted_grid[ok] / pdf_grid[ok]

        return delta_grid, cond_exp, max_residual

    # ------------------------------------------------------------------
    # PDF via FFT
    # ------------------------------------------------------------------

    def pdf(self, n=2**15, ell_max=200.0, n_coarse=400, n_polish=8):
        """Exact PDF via FFT of exp(phi(i*ell)).

        Parameters
        ----------
        n : int
            Number of FFT points (a power of 2 is fastest). Sets both the
            ell-space resolution (d_ell = 2*ell_max/n) and, via the
            uncertainty relation of the Fourier transform, the delta-space
            range and resolution of the returned grid.
        ell_max : float
            Half-width of the ell grid. Larger values resolve finer
            delta-space structure but push the continuation further from
            ell=0 -- inspect `max_residual` and any extrapolation warning
            when raising this.
        n_coarse : int
            Number of points in the cheap sequential continuation pass used
            to seed the fast vectorized polish (see phi_imag_axis).
        n_polish : int
            Number of vectorized Newton iterations applied to every point of
            the full FFT grid.

        Returns
        -------
        delta_grid, pdf_grid : ndarray
            Sorted delta grid (from the FFT frequencies) and the
            corresponding P(delta), NOT renormalized (check `pdf_grid`'s
            integral to gauge numerical accuracy -- see cross_check()).
        max_residual : float
            Diagnostic from phi_imag_axis; large values (>> 1e-6, say) mean
            ell_max/n_coarse/n_polish or order need adjusting.
        """
        ell = (np.arange(n) - n // 2) * (2.0 * ell_max / n)
        phi_vals, max_residual = self.phi_imag_axis(
            ell, n_coarse=n_coarse, n_polish=n_polish
        )
        integrand = np.exp(phi_vals)
        dell = ell[1] - ell[0]

        F = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(integrand)) * dell)
        delta_grid = np.fft.fftshift(np.fft.fftfreq(n, d=dell)) * 2.0 * np.pi
        pdf_grid = np.real(F) / (2.0 * np.pi)

        return delta_grid, pdf_grid, max_residual

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def cross_check(
        self, delta_grid, pdf_grid, delta_min=None, delta_max=None
    ):
        """Compare an FFT-derived PDF against the leading-order saddle-point
        formula in a "trusted" bulk region, as a sanity check.

        The saddle-point formula P(delta) ~ sqrt(Psi''/2pi) exp(-Psi) is
        exact to leading order in 1/sigma^2; genuine disagreement deep in
        the tails is expected and is the whole point of using this exact
        method there. But if the two disagree noticeably even close to the
        peak (small |delta|), something is off with the FFT/continuation
        settings (ell_max too small, order too low/high, insufficient
        n_coarse/n_polish) rather than genuine physics.

        Returns
        -------
        dict with keys 'delta', 'pdf_fft', 'pdf_saddle', 'rel_err',
        'n_nonconvex'
            restricted to [delta_min, delta_max] (defaults to the central
            +/- 1 unit of delta around the grid used to build this
            inverter, i.e. roughly the region where both methods should
            agree well). Silently clipped to the tabulated delta range this
            inverter was built from if you pass wider bounds explicitly
            (Psi is only known there; outside it the comparison would be
            meaningless spline extrapolation, not a real check). 'rel_err'
            is NaN wherever Psi is not locally convex (Psi''<=0) or too
            close to such a point for a reliable comparison -- pdf_saddle
            is simply undefined there, this is not a bug; 'n_nonconvex'
            counts how many of the returned points were excluded for this
            reason.
        """
        if self._orig_delta_grid is None:
            raise RuntimeError(
                "cross_check() needs a delta-space rate function to compare "
                "against, which is not available on an inverter built via "
                "CGFInverter.from_cgf() (only a tabulated CGF was supplied). "
                "Build the inverter via the default constructor (from a rate "
                "function) if you need this check."
            )
        delta_grid = np.asarray(delta_grid, dtype=float)
        pdf_grid = np.asarray(pdf_grid, dtype=float)

        grid_lo = float(np.min(self._orig_delta_grid))
        grid_hi = float(np.max(self._orig_delta_grid))

        if delta_min is None:
            delta_min = max(grid_lo, -1.0)
        if delta_max is None:
            delta_max = min(grid_hi, 1.0)

        if delta_min < grid_lo or delta_max > grid_hi:
            delta_min = max(delta_min, grid_lo)
            delta_max = min(delta_max, grid_hi)

        mask = (delta_grid >= delta_min) & (delta_grid <= delta_max)
        d = delta_grid[mask]

        spl = _UnivariateSpline(
            self._orig_delta_grid, self._orig_Psi_grid, s=0, k=4
        )
        Psi_d = spl(d)
        Psipp_d = spl.derivative(2)(d)
        valid = Psipp_d > 0
        pdf_saddle = np.full(d.shape, np.nan)
        pdf_saddle[valid] = np.sqrt(Psipp_d[valid] / (2 * np.pi)) * np.exp(
            -Psi_d[valid]
        )

        # Exclude a small buffer around any sign change of Psi'' too, since
        # pdf_saddle is numerically fragile there even where it technically
        # passes the Psi''>0 check (found this directly in testing).
        sign_changes = np.where(np.diff(np.sign(Psipp_d)) != 0)[0]
        near_sign_change = np.zeros(d.shape, dtype=bool)
        if sign_changes.size:
            buffer_pts = max(3, int(0.02 * len(d)))
            for i in sign_changes:
                lo = max(0, i - buffer_pts)
                hi = min(len(d), i + buffer_pts + 1)
                near_sign_change[lo:hi] = True

        rel_err = np.full(d.shape, np.nan)
        if np.any(valid):
            floor = 1e-3 * np.nanmax(pdf_saddle[valid])
            ok = valid & (pdf_saddle > floor) & (~near_sign_change)
        else:
            ok = np.zeros(d.shape, dtype=bool)
        rel_err[ok] = (
            np.abs(pdf_grid[mask][ok] - pdf_saddle[ok]) / pdf_saddle[ok]
        )

        return {
            "delta": d,
            "pdf_fft": pdf_grid[mask],
            "pdf_saddle": pdf_saddle,
            "rel_err": rel_err,
            "n_nonconvex": int(
                np.sum(~valid) + np.sum(near_sign_change & valid)
            ),
        }
