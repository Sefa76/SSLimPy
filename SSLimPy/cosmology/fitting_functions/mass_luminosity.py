"""
Calculate Mass-Luminosity relations for different models of line
emission.

All functions take a vector of masses in M_sun and return luminosities
in L_sun.

Model parameter values are given by a dictionary called self.model_par.  Each
function also takes a value of the redshift z even if the L(M) model is not
redshift dependent.  This allows the functions to be called consistently by
LineModel()

TODO:
Add in models from Matlab code
"""

import os

import astropy.units as u
import numpy as np

from scipy.interpolate import RegularGridInterpolator, interp1d


class mass_luminosity:

    def __init__(self, astro, model_par):
        self.astro = astro
        self.model_par = model_par

        self.SFR_folder = (
            "".join(os.path.dirname(os.path.realpath(__file__)).split("SSLimPy")[:-2])
            + "SSLimPy/SFR_tables/"
        )

        # Units and Redshifts
        self.Mpch = astro.Mpch
        self.Msunh = astro.Msunh

    def MassPow(self, Mvec, z):
        """
        Power law L(M)/L_sun = A*(M/M_sun)^b (See Breysse et al. 2015)

        Parameters:
        A         Overall amplitude, dimensionless
        b         Power law slope, dimensionless

        Assumed to be redshift independent

        >>> Mvec = np.array([1e10,1e11,1e12]) * u.Msun
        >>> self.model_par = {'A':2e-6, 'b':1.}
        >>> z = 3.0
        >>> print MassPow(Mvec,self.model_par,z)
        [   20000.   200000.  2000000.] solLum
        """
        z_grid = np.atleast_1d(z)[None, :]
        M_grid = np.atleast_1d(Mvec)[:, None]

        A = self.model_par["A"]
        b = self.model_par["b"]
        L = A * np.array(M_grid) ** b * u.Lsun * np.ones_like(z_grid)
        return np.squeeze(L)

    def DblPwr(self, Mvec, z):
        """
        Double power law with redshift dependence
        L(M)/Lsun = A * 10^(b1*z) * (M/1e8 Msun)^b2 * (1+M/M_*)^b3

        Parameters:
        A         Overall amplitude, dimensionless
        b1        Redshift slope, dimensionless
        b2        Low mass power law, dimensionless
        b3        High mass power law, dimensionless
        Mstar     Power law turnover mass, in M_sun

        >>> Mvec = np.array([1e10,1e11,1e12]) * u.Msun
        >>> self.model_par = {'A':5.8e-3, 'b1':0.35, 'b2':1.97, 'b3':-2.92, \
            'Mstar':8.e11*u.Msun}
        >>> z = 3.0
        >>> print DblPwr(Mvec,self.model_par,z)
        [    546.6...   37502...  462439...] solLum
        """
        z_grid = np.atleast_1d(z)[None, :]
        M_grid = np.atleast_1d(Mvec)[:, None]

        A = self.model_par["A"]
        b1 = self.model_par["b1"]
        b2 = self.model_par["b2"]
        b3 = self.model_par["b3"]
        Mstar = self.model_par["Mstar"]

        L = (
            A
            * np.power(10, b1 * z_grid)
            * (M_grid / (1.0e8 * u.Msun)) ** b2
            * (1.0 + M_grid / Mstar) ** b3
        ) * u.Lsun

        return np.squeeze(L)

    def CO_lines_scaling_LFIR(self, Mvec, z):
        """
        Returns the luminosity for CO lines lines that have empirical scaling relations with FIR luminosity

        Examples include: All the CO rotational ladder lines
        (From Kamenetzky+2016, arXiv:1508.05102)

        Relation is: log10(LFIR) = alpha*log10(LCO')+beta

        Parameters:
            -SFR:       SFR of the halo in Msun/yr
            -pars:      Dictionary of parameters for the model
                -alpha
                -beta
        """
        z_grid = np.atleast_1d(z)[None, :]
        M_grid = np.atleast_1d(Mvec)[:, None]

        alpha = self.model_par["alpha"]
        beta = self.model_par["beta"]
        SFR_file = self.model_par["SFR_file"]

        # Read and interpolate Behroozi SFR(M) data
        SFR = self.get_SFR(Mvec, z, SFR_file)
        # is there quenching?
        try:
            do_quench = self.model_par["do_quench"]
            if do_quench:
                fQint = self.process_fq()
                SFR *= 1 - fQint((np.log10(M_grid.to(u.Msun).value), 1 + z_grid))
        except:
            pass

        # Compute IR luminosity in Lsun from Kennicutt 1998, arXiv:9807187
        LIR = (SFR * (1 / 4.5e-44) * u.erg / u.s).to(u.Lsun)
        Lp = np.power(10, ((np.log10(LIR.value) - beta) / alpha))
        L = (4.9e-5 * u.Lsun) * Lp * (self.astro.nu[None, :] / (115.27 * u.GHz)) ** 3

        return np.squeeze(L)

    def COMAP_Fid(self, Mvec, z):
        """
        New COMAP fiducial double-power law model
        L(M) = C/((M/Ms)^A+(M/Ms)^B)

        Parameters:
        A   Low-mass slope
        B   High-mass slope
        C   Overall normalization
        Ms  Turnover mass

        Predicted values for (A, B, log C, log (Ms/Msol), sigma):
        * pessimistic: (-3.7, 7.0, 11.1, 12.5, 0.36)
        * realistic: (-2.75, 0.05, 10.61, 12.3, 0.42)
        * realistic-plus: (-2.85, -0.42, 10.63, 12.3, 0.42)
        * optimistic: (-2.4, -0.5, 10.45, 12.21, 0.36))
        """
        z_grid = np.atleast_1d(z)[None, :]
        M_grid = np.atleast_1d(Mvec)[:, None]

        A = self.model_par["A"]
        B = self.model_par["B"]
        C = self.model_par["C"]
        Ms = self.model_par["Ms"]

        L = C / ((M_grid / Ms) ** A + (M_grid / Ms) ** B) * np.ones_like(z_grid)
        L *= (
            4.9e-5
            * u.Lsun
            * (np.atleast_1d(self.astro.nu)[None, :] / (115.27 * u.GHz)) ** 3
        )
        return np.squeeze(L)

    def TonyLi(self, Mvec, z):
        """
        CO emission model from Li et al. (2016).  Uses Behroozi et al. SFR(M)
        results.

        NOTE ON THIS MODEL: The Li et al. model has two types of scatter: one on
        SFR(M) and one on LCO(SFR), denoted as sigma_SFR and sigma_LCO.  The LCO
        scatter should be entered into LineModel() as the usual sigma_scatter
        input.  However, the SFR scatter behaves differently in that it does not
        preserve mean(LCO), but preserves mean(SFR) instead.  Thus it should be
        given as part of self.model_par, there are specific hacks added to LineModel() to
        account for this.

        Parameters:
        alpha         Slope of logLIR/logLCO relation, dimensionless
        beta          Intercept of logLIR/logLCO relation, dimensionless
        dMF           10^10 times SFR/LIR normalization (See Li et al. Eq 1),
                        dimensionless
        SFR_file  Filename where Behroozi et al. data is stored, default
                        'sfr_release.dat'. File can be downloaded from
                        peterbehroozi.com/data, (string)
        Mcut_min  Minimum mass below which L=0 (in M_sun)
        Mcut_max  Maximum mass above which L=0 (in M_sun)

        >>> Mvec = np.array([1e10,1e11,1e12]) * u.Msun
        >>> self.model_par = {'alpha':1.17, 'beta':0.21, 'dMF':1.0,\
            'BehrooziFile':'sfr_release.dat'}
        >>> z = 3.0
        >>> print TonyLi(Mvec,self.model_par,z)
        [  2.05...e+02   7.86...e+03   4.56...e+05] solLum
        """
        z_grid = np.atleast_1d(z)[None, :]
        M_grid = np.atleast_1d(Mvec)[:, None]

        alpha = self.model_par["alpha"]
        beta = self.model_par["beta"]
        dMF = self.model_par["dMF"]
        SFR_file = self.model_par["SFR_file"]

        # Read and interpolate Behroozi SFR(M) data
        SFR = self.get_SFR(Mvec, z, SFR_file)
        # is there quenching?
        try:
            do_quench = self.model_par["do_quench"]
            if do_quench:
                fQint = self.process_fq()
                SFR *= 1 - fQint((np.log10(M_grid.to(u.Msun).value), 1 + z_grid))
        except:
            pass

        # Compute IR luminosity in Lsun
        LIR = SFR / (dMF * 1e-10)

        # Compute L'_CO in K km/s pc^2
        Lprime = (10.0**-beta * LIR) ** (1.0 / alpha)

        # Compute LCO
        L = (
            (4.9e-5 * u.Lsun)
            * Lprime
            * (np.atleast_1d(self.astro.nu)[None, :] / (115.27 * u.GHz)) ** 3
        )

        return np.squeeze(L)

    def SilvaCII(self, Mvec, z):
        """
        Silva et al. (2015) CII model, relates CII luminosity and SFR by
        log10(L_CII/Lsun) = a_LCII*log10(SFR/(Msun/yr)) + b_LCII

        SFR(M) computed from the double power law fit in their Eq. (8), with
        parameters interpolated from their Table 2.

        Note that the L(M) relations derived from this model are a variant on the
        DblPwr model above, but with the input parameters changed to match the
        Silva et al. numbers

        Parameters:
        a   a_LCII parameter in L(SFR), dimensionless
        b   b_LCII parameter in L(SFR)

        >>> Mvec = np.array([1e10,1e11,1e12]) * u.Msun
        >>> self.model_par = {'a':0.8475, 'b':7.2203}
        >>> z = 7.5
        >>> print SilvaCII(Mvec,self.model_par,z)
        [  4.58...e+06   1.61...e+08   6.89...e+08] solLum
        """
        z_grid = np.atleast_1d(z)[None, :]
        M_grid = np.atleast_1d(Mvec)[:, None]

        aLCII = self.model_par["a"]
        bLCII = self.model_par["b"]
        SFR_file = self.model_par["SFR_file"]

        SFR = self.get_SFR(Mvec, z, SFR_file)
        # is there quenching?
        try:
            do_quench = self.model_par["do_quench"]
            if do_quench:
                fQint = self.process_fq()
                SFR *= 1 - fQint((np.log10(M_grid.to(u.Msun).value), 1 + z_grid))
        except:
            pass

        # LCII relation
        L = 10 ** (aLCII * np.log10(SFR / (1 * u.Msun / u.yr)) + bLCII) * u.Lsun

        return np.squeeze(L)

    def FonsecaLyalpha(self, Mvec, z):
        """
        Fonseca et al. 2016 model for Lyman alpha emission line. Relates Lyman alpha
        luminosity by
        L_Lya [erg/s] = K_Lyalpha * 1e+41 * SFR, eq. 8
        assuming a triple power law model for SFR, eq. 11, with
            fit parameters in Table 1

        Parameters:

        Aext         dust extinction
        fLyaesc      fraction of Lyman alpha photons that escape the galaxy
        RLya         constant relating SFR with Luminosity *1e42 erg/s
        fUVesc       fraction of UV that escape the galaxy

        SFR_file     file with SFR
        """
        z_grid = np.atleast_1d(z)[None, :]
        M_grid = np.atleast_1d(Mvec)[:, None]

        RLya = self.model_par["RLya"] * 1e42 * u.erg / u.s * (u.Msun / u.yr) ** -1
        SFR_file = self.model_par["SFR_file"]
        Aext = self.model_par["Aext"]
        fLyaesc = self.model_par["fLyaesc"]
        fUVesc = self.model_par["fUVesc"]

        SFR = self.get_SFR(Mvec, z, SFR_file)
        # is there quenching?
        try:
            do_quench = self.model_par["do_quench"]
            if do_quench:
                fQint = self.process_fq()
                SFR *= 1 - fQint((np.log10(M_grid.to(u.Msun).value), 1 + z_grid))
        except:
            pass

        fUVdust = 10 ** (-Aext / 2.5)
        K_Lyalpha = (fUVdust - fUVesc) * fLyaesc * RLya

        L = (SFR * K_Lyalpha).to(u.Lsun)
        return np.squeeze(L)

    def SilvaLyalpha_12(self, Mvec, z):
        """
        Silva et al. 2012 model for Lyman alpja emission line. Has a relation between
        L and SFR depending on z (interpolates over values).

        Parameters:

        SFR_file file with SFR
        """
        z_grid = np.atleast_1d(z)[None, :]
        M_grid = np.atleast_1d(Mvec)[:, None]

        # Get SFR file
        SFR_file = self.model_par["SFR_file"]

        SFR = self.get_SFR(Mvec, z, SFR_file)
        # is there quenching?
        try:
            do_quench = self.model_par["do_quench"]
            if do_quench:
                fQint = self.process_fq()
                SFR *= 1 - fQint((np.log10(M_grid.to(u.Msun).value), 1 + z_grid))
        except:
            pass

        # fraction of Lya photons not absorbed by dust
        Cdust = 3.34
        xi = 2.57
        fLy = Cdust * 1e-3 * (1.0 + z_grid) ** xi
        # escape fraction of ionizing radiation
        zint = np.array([10.4, 8.2, 6.7, 5.7])
        alphaint = np.array([27.8, 13.0, 5.18, 3.42]) * 1e-3
        betaint = np.array([0.105, 0.179, 0.244, 0.262])
        alpha = 10 ** interp1d(
            zint,
            np.log10(alphaint),
            bounds_error=False,
            fill_value="extrapolate",
            kind="cubic",
        )(z_grid)
        beta = 10 ** interp1d(
            zint,
            np.log10(betaint),
            bounds_error=False,
            fill_value="extrapolate",
            kind="cubic",
        )(z_grid)
        fesc = np.exp(-alpha * M_grid.to(u.Msun).value ** beta)
        # Luminosity due to recombinations
        Lrec = 1.55e42 * (1.0 - fesc) * fLy * SFR / (u.Msun / u.yr) * (u.erg / u.s)
        # Luminosity due to excitation
        Lexc = 4.03e41 * (1.0 - fesc) * fLy * SFR / (u.Msun / u.yr) * (u.erg / u.s)
        # Luminosity due to gas cooling
        Lcool = (
            1.69e35
            * fLy
            * (
                (1.0 + M_grid.to(u.Msun).value / 1e8)
                * (1.0 + M_grid.to(u.Msun).value) ** 2.1
                * (1.0 + M_grid.to(u.Msun).value) ** -3
            )
            * (u.erg / u.s)
        )
        # Luminosity from continuum emission
        Lstellar = 5.12e40
        Lfreefree = 1.1e35
        Lfreebound = 1.47e37
        L2phot = 2.41e38
        Lcont = (
            (Lstellar + Lfreefree + Lfreebound + L2phot)
            * fLy
            * SFR
            / (u.Msun / u.yr)
            * (u.erg / u.s)
        )

        L = (Lrec + Lexc + Lcool + Lcont).to(u.Lsun)

        return np.squeeze(L)

    def Chung_Lyalpha(self, Mvec, z):
        """
        Model for Lyman-alpha line used in Chung+2019 (arXiv:1809.04550)
        Model f esc and L dependent on SFR

        Parameters:
            SFR_file     file with SFR
            -C      Conversion between SFR and Ly-alpha luminosity
            -xi, zeta, psi, z0, f0, SFR0    Parametrize the escape fraction,
                                            reflecting the possibility of photons
                                            being absorbed by dust
        """
        z_grid = np.atleast_1d(z)[None, :]
        M_grid = np.atleast_1d(Mvec)[:, None]

        C = self.model_par["C"]
        xi = self.model_par["xi"]
        zeta = self.model_par["zeta"]
        psi = self.model_par["psi"]
        z0 = self.model_par["z0"]
        f0 = self.model_par["f0"]
        SFR0 = self.model_par["SFR0"]
        SFR_file = self.model_par["SFR_file"]

        SFR = self.get_SFR(Mvec, z, SFR_file)
        # is there quenching?
        try:
            do_quench = self.model_par["do_quench"]
            if do_quench:
                fQint = self.process_fq()
                SFR *= 1 - fQint((np.log10(M_grid.to(u.Msun).value), 1 + z_grid))
        except:
            pass

        fesc = (
            ((1 + np.exp(-xi * (z_grid - z0))) ** (-zeta))
            * (f0 + ((1 - f0) / (1 + (SFR / SFR0) ** (psi))))
        ) ** 2
        L = (C * SFR * fesc * u.erg / u.s).to(u.Lsun)

        return np.squeeze(L)

    def KSrel(self, Mvec, z):
        """
        Uses the Kennicutt-Schmidt relation: linear with SFR, adding an
        absorption term.
        L = K*SFR*10^{-Aext/2.5}
        Parameters:
            K_Halpha     normalization between SFR and L -> With units!!
            Aext         Extinction
            SFR_file     file with SFR
        """
        z_grid = np.atleast_1d(z)[None, :]
        M_grid = np.atleast_1d(Mvec)[:, None]

        K = self.model_par["K"]
        Aext = self.model_par["Aext"]
        SFR_file = self.model_par["SFR_file"]

        SFR = self.get_SFR(Mvec, z, SFR_file)
        # is there quenching?
        try:
            do_quench = self.model_par["do_quench"]
            if do_quench:
                fQint = self.process_fq()
                SFR *= 1 - fQint((np.log10(M_grid.to(u.Msun).value), 1 + z_grid))
        except:
            pass

        L = (SFR * K * 10 ** (-Aext / 2.5)).to(u.Lsun)
        return np.squeeze(L)

    def GongHalpha(self, Mvec, z):
        """
        Gong et al. 2017 model for Halpha emission line. Relates Halpha
        luminosity by
        L_Halpha [erg/s] = K_Halpha * 1e+41 * SFR, eq. 3
        assuming a doble power law model for SFR, eq. 6, with
            fit parameters in Table 1

        Parameters:
        K_Halpha     normalization between SFR and L
        Aext         Extinction
        SFR_file     file with SFR
        """
        z_grid = np.atleast_1d(z)[None, :]
        M_grid = np.atleast_1d(Mvec)[:, None]

        K_Halpha = (
            self.model_par["K_Halpha"] * 1e41 * u.erg / u.s * (u.Msun / u.yr) ** -1
        )
        Aext = self.model_par["Aext"]
        SFR_file = self.model_par["SFR_file"]

        SFR = self.get_SFR(Mvec, z, SFR_file)
        # is there quenching?
        try:
            do_quench = self.model_par["do_quench"]
            if do_quench:
                fQint = self.process_fq()
                SFR *= 1 - fQint((np.log10(M_grid.to(u.Msun).value), 1 + z_grid))
        except:
            pass

        L = (SFR * K_Halpha * 10 ** (-Aext / 2.5)).to(u.Lsun)
        return np.squeeze(L)

    def GongHbeta(self, Mvec, z):
        """
        Gong et al. 2017 model for Hbeta emission line. Relates Hbeta
        luminosity by
        L_Hbeta [erg/s] = K_Hbeta * 1e+41 * SFR, eq. 3
        assuming a doble power law model for SFR, eq. 6, with
            fit parameters in Table 1

        Parameters:
        K_Hbeta     normalization between SFR and L
        Aext         Extinction
        SFR_file     file with SFR
        """
        z_grid = np.atleast_1d(z)[None, :]
        M_grid = np.atleast_1d(Mvec)[:, None]

        K_Hbeta = self.model_par["K_Hbeta"] * 1e41 * u.erg / u.s * (u.Msun / u.yr) ** -1
        Aext = self.model_par["Aext"]
        SFR_file = self.model_par["SFR_file"]

        SFR = self.get_SFR(Mvec, z, SFR_file)
        # is there quenching?
        try:
            do_quench = self.model_par["do_quench"]
            if do_quench:
                fQint = self.process_fq()
                SFR *= 1 - fQint((np.log10(M_grid.to(u.Msun).value), 1 + z_grid))
        except:
            pass

        L = (SFR * K_Hbeta * 10 ** (-Aext / 2.5)).to(u.Lsun)
        return np.squeeze(L)

    def GongOIII(self, Mvec, z):
        """
        Gong et al. 2017 model for OIII emission line. Relates OIII
        luminosity by
        L_OIII [erg/s] = K_OIII * 1e+41 * SFR, eq. 3
        assuming a doble power law model for SFR, eq. 6, with
            fit parameters in Table 1

        Parameters:
        K_OIII     normalization between SFR and L
        Aext         Extinction
        SFR_file     file with SFR
        """
        z_grid = np.atleast_1d(z)[None, :]
        M_grid = np.atleast_1d(Mvec)[:, None]

        K_OIII = self.model_par["K_OIII"] * 1e41 * u.erg / u.s * (u.Msun / u.yr) ** -1
        Aext = self.model_par["Aext"]
        SFR_file = self.model_par["SFR_file"]

        SFR = self.get_SFR(Mvec, z, SFR_file)
        # is there quenching?
        try:
            do_quench = self.model_par["do_quench"]
            if do_quench:
                fQint = self.process_fq()
                SFR *= 1 - fQint((np.log10(M_grid.to(u.Msun).value), 1 + z_grid))
        except:
            pass

        L = (SFR * K_OIII * 10 ** (-Aext / 2.5)).to(u.Lsun)
        return np.squeeze(L)

    def GongOII(self, Mvec, z):
        """
        Gong et al. 2017 model for OII emission line. Relates OII
        luminosity by
        L_OII [erg/s] = K_OII * 1e+41 * SFR, eq. 3
        assuming a doble power law model for SFR, eq. 6, with
            fit parameters in Table 1

        Parameters:
        K_OII     normalization between SFR and L
        Aext         Extinction
        SFR_file     file with SFR
        """
        z_grid = np.atleast_1d(z)[None, :]
        M_grid = np.atleast_1d(Mvec)[:, None]

        K_OII = self.model_par["K_OII"] * 1e41 * u.erg / u.s * (u.Msun / u.yr) ** -1
        Aext = self.model_par["Aext"]
        SFR_file = self.model_par["SFR_file"]

        SFR = self.get_SFR(Mvec, z, SFR_file)
        # is there quenching?
        try:
            do_quench = self.model_par["do_quench"]
            if do_quench:
                fQint = self.process_fq()
                SFR *= 1 - fQint((np.log10(M_grid.to(u.Msun).value), 1 + z_grid))
        except:
            pass

        L = (SFR * K_OII * 10 ** (-Aext / 2.5)).to(u.Lsun)
        return np.squeeze(L)

    def HI_lowz_Villaescusa(self, Mvec, z):
        """
        Fitting function to M_HI(M_h) from Villaescusa-Navarro et al. 2018
        using their FOF halos.
        Note that mass quantities are given in Msun/h

        M_HI(M,z) = M_0*(M/Mmin)^z*exp(-(Mmin/M)^0.35)
        """
        z_grid = np.atleast_1d(z)[None, :]
        M_grid = np.atleast_1d(Mvec)[:, None]

        zint = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        alphaint = np.array([0.24, 0.53, 0.60, 0.76, 0.79, 0.74])
        M0int = np.array([4.3, 1.5, 1.3, 0.29, 0.14, 0.19]) * 1e10
        Mminint = np.array([20.0, 6.0, 3.6, 0.67, 0.21, 0.20]) * 1e11
        Mmaxint = np.array([35.0, 10.0, 3.0, 1.6, 0.63, 0.26]) * 1e13

        alpha = interp1d(
            zint, alphaint, kind="linear", bounds_error=False, fill_value=alphaint[-1]
        )(z_grid)
        M0 = (
            interp1d(
                zint, M0int, kind="linear", bounds_error=False, fill_value=M0int[-1]
            )(z_grid)
            * self.Msunh
        ).to(u.Msun)
        Mmin = (
            interp1d(
                zint, Mminint, kind="linear", bounds_error=False, fill_value=Mminint[-1]
            )(z_grid)
            * self.Msunh
        ).to(u.Msun)
        Mmax = (
            interp1d(
                zint, Mmaxint, kind="linear", bounds_error=False, fill_value=Mmaxint[-1]
            )(z_grid)
            * self.Msunh
        ).to(u.Msun)

        M_HI = M0 * (M_grid / Mmin) ** alpha * np.exp(-((Mmin / M_grid) ** 0.35))
        M_HI[Mvec[:, None] >= Mmax] = 1e-44 * M_HI.unit

        CLM = 6.25e-9 * u.Lsun / u.Msun  # Conversion factor btw MHI and LHI
        L = CLM * M_HI
        return np.squeeze(L)

    def MHI_21cm_Obuljen(self, Mvec, z):
        """
        Obuljen et al. (2018) 21cm MHI(M) model, relates MHI to halo mass by
        MHI = M0 * (M/Mmin)^alpha * exp(-Mmin/M)

        NOTE that the best fit values given by Obuljen et al. for M0 and Mmin are
        in Msun/h units

        Parameters
        M0      Overall normalization of MHI(M) (in Msun)
        Mmin    Location of low-mass exponential cutoff (in Msun)
        alpha   Slope at high-mass (dimensionless)

        >>> Mvec = np.array([1e10,1e11,1e12]) * u.Msun
        >>> self.model_par = {'M0':4.73e8*u.Msun,'Mmin':2.66e11*u.Msun,'alpha':0.44}
        >>> z = 0.03
        >>> print MHI_21cm(Mvec,self.model_par,z)
        [  1.94...e-12   1.33...e-01   4.03...e+00] solLum
        """
        z_grid = np.atleast_1d(z)[None, :]
        M_grid = np.atleast_1d(Mvec)[:, None]

        M0 = self.model_par["M0"]
        Mmin = self.model_par["Mmin"]
        alpha = self.model_par["alpha"]

        CLM = 6.25e-9 * u.Lsun / u.Msun  # Conversion factor btw MHI and LHI

        MHI = M0 * (M_grid / Mmin) ** alpha * np.exp(-Mmin / M_grid)
        L = CLM * MHI * np.ones_like(z_grid)
        return np.squeeze(L)

    def MHI_21cm_Padmanabhan(self, Mvec, z):
        """
        Padmamabhan & Kulkarni 2017 21cm MHI(M) model, relates MHI to halo mass by
        MHI = 2*N*M/((M/M1)^-b1+(M/M1)^y1)

        where N, M0, b and y are free parameters fit to observations as function
        of redshift. We use the best fit here
        """
        z_grid = np.atleast_1d(z)[None, :]
        M_grid = np.atleast_1d(Mvec)[:, None]

        # Values at z=0
        M10 = 4.58e11 * u.Msun
        N10 = 9.89e-3
        b10 = 0.9
        y10 = 0.74
        # z evolution params
        M11 = 1.56
        N11 = 0.009
        b11 = -1.08
        y11 = 4.07
        # z evolution
        M1 = 10 ** (np.log10(M10 / u.Msun) + z_grid / (z_grid + 1) * M11) * u.Msun
        N1 = N10 + z_grid / (z_grid + 1) * N11
        b1 = b10 + z_grid / (z_grid + 1) * b11
        y1 = y10 + z_grid / (z_grid + 1) * y11

        MHI = 2 * N1 * M_grid * ((M_grid / M1) ** -b1 + (M_grid / M1) ** y1) ** -1

        CLM = 6.25e-9 * u.Lsun / u.Msun  # Conversion factor btw MHI and LHI
        L = CLM * MHI
        return np.squeeze(L)

    def Constant_L(self, Mvec, z):
        """
        Model where every halo has a constant luminosity independent of mass.
        Still has cutoffs at Mcut_min and Mcut_max.

        Intended primarily for sanity checks and debugging.

        Parameters:
        L0  Luminosity of every halo

        >>> Mvec = np.array([1e10,1e11,1e12]) * u.Msun
        >>> self.model_par = {'L0':1*u.Lsun}
        >>> z = 1
        >>> print Constant_L(Mvec,self.model_par,z)
        [ 1.  1.  1.] solLum
        """
        z = np.atleast_1d(z)
        Mvec = np.atleast_1d(Mvec)

        L = self.model_par["L0"] * np.ones((Mvec.size, z.size))

        return np.squeeze(L)

    ###################
    # Other functions #
    ###################

    def get_SFR(self, M, z, SFR_file):
        """
        Returns SFR(M,z) interpolated from a file (either a M,z,SFR table
        or a table of fit parameters to a SFR(M,z) relation). It takes which
        kind of case is from the name of the SFR_file: when a specific case must
        be use, include that in the name of the file.

        SFR must be in Msun/year and halo mass in Msun
        """

        if "Silva" in SFR_file:
            # Interpolate parameters for SFR from Table 2 of Silva et al. 2015
            SFR = self.Silva_SFR(M, z, SFR_file)
        elif "Fonseca" in SFR_file:
            # Interpolate parameters for SFR from Table 1 of Fonseca et al. 2016
            SFR = self.Fonseca_SFR(M, z, SFR_file)
        elif "Gong" in SFR_file:
            # Interpolate parameters for SFR from Table 1 of Gong et al. 2016
            SFR = self.Gong_SFR(M, z, SFR_file)
        else:
            SFR = self.SFR_Mz_2dinterp(M, z, SFR_file)

        return SFR

    def SFR_Mz_2dinterp(self, M, z, SFR_file):
        """
        Returns SFR(M,z) interpolated from tables of 1+z, log10(Mhalo/Msun) and
        log10(SFR / (Msun/yr)), in three columns, where 1+z is the innermost index
        (the one running fast compared with the mass)
        """
        z_grid = np.atleast_1d(z)[None, :]
        M_grid = np.atleast_1d(M)[:, None]

        try:
            x = np.loadtxt(self.SFR_folder + SFR_file)
        except:
            x = np.loadtxt(SFR_file)
        zb = np.unique(x[:, 0]) - 1.0
        logMb = np.unique(x[:, 1])
        logSFRb = x[:, 2].reshape(len(zb), len(logMb), order="F")

        logSFR_interp = RegularGridInterpolator(
            (zb, logMb), logSFRb, bounds_error=False, fill_value=-40.0
        )

        logM_grid = np.log10((M_grid.to(u.Msun)).value)
        SFR = np.power(10, logSFR_interp((z_grid, logM_grid))) * u.Msun / u.yr

        return np.squeeze(SFR)

    def Silva_SFR(self, M, z, SFR_file):
        """
        Returns SFR(M,z) interpolated from values in Table 2 of Silva et al. 2015
        """
        z_grid = np.atleast_1d(z)[None, :]
        M_grid = np.atleast_1d(M)[:, None]

        try:
            x = np.loadtxt(self.SFR_folder + SFR_file)
        except:
            x = np.loadtxt(SFR_file)
        z0 = x[0, :]
        M0 = (
            10
            ** interp1d(
                z0, np.log10(x[1, :]), bounds_error=False, fill_value="extrapolate"
            )(z_grid)
            * u.Msun
            / u.yr
        )
        Ma = (
            interp1d(z0, x[2, :], bounds_error=False, fill_value="extrapolate")(z_grid)
            * u.Msun
        )
        Mb = (
            interp1d(z0, x[3, :], bounds_error=False, fill_value="extrapolate")(z_grid)
            * u.Msun
        )
        a = interp1d(z0, x[4, :], bounds_error=False, fill_value="extrapolate")(z_grid)
        b = interp1d(z0, x[5, :], bounds_error=False, fill_value="extrapolate")(z_grid)

        SFR = M0 * (M_grid / Ma) ** a * (1 + M_grid / Mb) ** b
        return np.squeeze(SFR)

    def Gong_SFR(self, M, z, SFR_file):
        """
        Returns SFR(M,z) interpolated from values in Table 1 of Gong et al. 2016
        """
        z = np.atleast_1d(z)
        M = np.atleast_1d(M)
        M_grid = M[:, None]
        z_grid = z[None, :]

        SFR = np.zeros((len(M), len(z)))

        z_mid = np.where((z < 5) and (z >= 4))
        z_low = np.where(z < 4)
        Mlim_mid = 1e12 * u.Msun
        Mlim_low = 1e13 * u.Msun

        try:
            x = np.loadtxt(self.SFR_folder + SFR_file)
        except:
            x = np.loadtxt(SFR_file)
        z0 = x[:, 0]
        a = interp1d(z0, x[:, 1], bounds_error=False, fill_value="extrapolate")(z_grid)
        b = interp1d(z0, x[:, 2], bounds_error=False, fill_value="extrapolate")(z_grid)
        c = interp1d(z0, x[:, 3], bounds_error=False, fill_value="extrapolate")(z_grid)

        M1 = 1e8 * u.Msun
        M2 = 4e11 * u.Msun

        M_mid = np.minimum(M_grid, Mlim_mid)
        SFR[:, z_mid] = (
            np.power(10.0, a[z_mid])
            * np.power(M_mid / M1, b[z_mid])
            * np.power(1.0 + M_mid / M2, c[z_mid])
            * u.Msun
            / u.yr
        )
        M_low = np.minimum(M_grid, Mlim_low)
        SFR[:, z_low] = (
            np.power(10.0, a[z_low])
            * np.power(M_low / M1, b[z_low])
            * np.power(1.0 + M_low / M2, c[z_low])
            * u.Msun
            / u.yr
        )

        return np.squeeze(SFR)

    def Fonseca_SFR(self, M, z, SFR_file):
        """
        Returns SFR(M,z) interpolated from values in Table 1 of Fonseca et al. 2016
        """
        M_grid = np.atleast_1d(M)[:, None]
        z_grid = np.atleast_1d(z)[None, :]

        try:
            x = np.loadtxt(self.SFR_folder + SFR_file)
        except:
            x = np.loadtxt(SFR_file)
        z0 = x[:, 0]
        M0 = interp1d(z0, x[:, 1], bounds_error=False, fill_value="extrapolate")(z_grid)
        Mb = (
            interp1d(z0, x[:, 2], bounds_error=False, fill_value="extrapolate")(z_grid)
            * u.Msun
        )
        Mc = (
            interp1d(z0, x[:, 3], bounds_error=False, fill_value="extrapolate")(z_grid)
            * u.Msun
        )
        a = interp1d(z0, x[:, 4], bounds_error=False, fill_value="extrapolate")(z_grid)
        b = interp1d(z0, x[:, 5], bounds_error=False, fill_value="extrapolate")(z_grid)
        c = interp1d(z0, x[:, 6], bounds_error=False, fill_value="extrapolate")(z_grid)

        Ma = 1e8 * u.Msun

        SFR = (
            M0
            * (M_grid / Ma) ** a
            * (1.0 + M_grid / Mb) ** b
            * (1 + M_grid / Mc) ** c
            * u.Msun
            / u.yr
        )

        return np.squeeze(SFR)

    def process_fq(self) -> RegularGridInterpolator:
        """
        Returns a 2D intepolated fQ from tables of Universe Machine. The table is 1+z, log10(Mhalo/Msun) and fQ
        """

        files = os.listdir(self.SFR_folder + "qf/")
        a = []
        mat = np.zeros((32, len(files)))

        counter = 0
        for name in files:
            a.append(float((name.split("qf_hm_a")[1]).split(".dat")[0]))

        inds = np.argsort(np.array(a))
        a = np.sort(np.array(a))

        names = []
        for i in range(len(inds)):
            names.append(files[inds[i]])

        for ia in range(len(a)):
            data = np.genfromtxt(self.SFR_folder + "qf/" + names[ia])
            mat[:, ia] = data[:, 1]

        logMh = data[:, 0]
        zp1 = 1 / a

        fQ_interp = RegularGridInterpolator((logMh, zp1), mat, bounds_error=True)

        return fQ_interp

    if __name__ == "__main__":
        import doctest

        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
