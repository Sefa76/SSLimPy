import sys
import types
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
import astropy.units as u
import astropy.constants as c

from copy import deepcopy

sys.path.append("../")
from SSLimPy.interface import config as cfg
from SSLimPy.cosmology import cosmology
from SSLimPy.cosmology.fitting_functions import bias_fitting_functions as bf
from SSLimPy.cosmology.fitting_functions import halo_mass_functions as HMF
from SSLimPy.cosmology.fitting_functions import luminosity_functions as lf
from SSLimPy.cosmology.fitting_functions import mass_luminosity as ml

class astro_functions:
    def __init__(self, cosmopars=dict(), astropars=dict()):

        self.astroparams = deepcopy(astropars)
        self.set_astrophysics_defaults()

        self.astrotracer = self.astroparams["astro_tracer"]

        ### TEXT VOMIT ###
        if cfg.settings["verbosity"]>1:
            self.recap_astro()
        ##################

        if cosmopars:
            self.cosmopars = deepcopy(cfg.fiducialcosmoparams)
        else:
            self.cosmopars = deepcopy(cosmopars)

        if self.cosmopars == cfg.fiducialcosmoparams:
            # No need to recompute the cosmology
            self.cosmology = cfg.fiducialcosmo
        else:
            self.cosmology = cosmology.cosmo_functions(cosmopars, cfg.input_type)

        # Current units
        self.hubble = self.cosmology.h()
        self.Mpch = u.Mpc / self.hubble
        self.Msunh = u.Msun / self.hubble
        self.rho_crit = 2.77536627e11*(self.Msunh*self.Mpch**-3).to(u.Msun*u.Mpc**-3) #Msun/Mpc^3

        # Internal samples for computations
        self.M = np.geomspace(self.astroparams["Mmin"],self.astroparams["Mmax"],self.astroparams["nM"])
        self.L = np.geomspace(self.astroparams["Lmin"],self.astroparams["Lmax"],self.astroparams["nL"])
        # find the redshifts for fequencies asked for:
        self.nu = self.astroparams["nu"]
        self.nuObs = self.astroparams["nuObs"]
        self.z = np.atleast_1d((self.astroparams["nu"] / self.astroparams["nuObs"]).to(1).value - 1)

        self.sigmaM, self.dsigmaM_dM = self.create_sigmaM_funcs()

        # save the astro results
        self.results = types.SimpleNamespace()
        self.results.sigmaM = self.sigmaM(self.M,self.z)
        self.results.dsigmaM_dM = self.dsigmaM_dM(self.M,self.z)

        # Check passed models
        self.init_model()
        self.init_halo_mass_function()
        self.init_bias_function()

        # bias function
        # !Without Corrections for nongaussianity!
        self.delta_crit = 1.686
        self.b_of_M = getattr(self.bias_function,self.astroparams["bias_model"])
        self.results.b_of_M = self.b_of_M(dc=self.delta_crit,nu=self.delta_crit/self.results.sigmaM)

        # halo mass function
        M_input = self.M.to(self.Msunh)
        rho_input = 2.77536627e11* self.cosmology.Omega(0,self.astrotracer)*(self.Msunh*self.Mpch**-3).to(self.Msunh*self.Mpch**-3)
        self.dn_dM_of_M = getattr(self.halo_mass_function,self.astroparams["hmf_model"])
        self.results.dn_dM_of_M = self.dn_dM_of_M(M_input,rho_input,self.z)

        if "ML" in self.astroparams["model_type"]:
            self.L_of_M  = getattr(self.mass_luminosity_function,self.astroparams["model_name"])
            self.results.L_of_M = self.L_of_M(self.M.to(u.Msun),self.z)

            sigma = np.maximum(cfg.settings["sigma_scatter"],0.05)
            # Special case for Tony Li model- scatter does not preserve LCO
            if self.model_name=='TonyLi':
                alpha = self.model_par['alpha']
                sig_SFR = self.model_par['sig_SFR']
                #assume sigma and sig_SFR are totally uncorrelated
                sigma = (sigma**2 + sig_SFR**2/alpha**2)**0.5
            sigma_base_e = sigma*2.302585

            L_of_M_and_z = np.reshape(self.results.L_of_M,(*self.M.shape,*self.z.shape))
            dn_dM_of_M_and_z = np.reshape(self.dn_dM_of_M,(*self.M.shape,*self.z.shape))
            flognorm =  self.lognormal(self.L[None,:,None],np.log(L_of_M_and_z.value)[:,None,:]-0.5*sigma_base_e**2.,sigma_base_e)
            CFL = flognorm * dn_dM_of_M_and_z[:,None,:]
            self.results.dn_dL_of_L = np.squeeze(np.trapz(CFL,self.M,axis=0))

        elif "LF" in self.astroparams["model_type"]:
            LF_par = {'A':1.,'b':1.,'Mcut_min':self.astroparams["Mmin"],'Mcut_max':self.astroparams["Mmax"]}
            off_mass_luminosity = ml.mass_luminosity(self,LF_par)
            self.L_of_M = getattr(off_mass_luminosity,'MassPow')
            self.results.L_of_M = self.L_of_M(self.M.to(u.Msun),self.z)

            self.dn_dL_of_L = getattr(self.luminosity_function,self.astroparams["model_name"])
            self.results.dn_dL_of_L = self.dn_dL_of_L(self.L.to(u.Lsun))

    def sigmaM_of_z(self, M, z):
        '''
        Mass (or cdm+b) variance at target redshift
        '''
        tracer = self.astrotracer

        rhoM = self.rho_crit*self.cosmology.Omega(0,tracer)
        R = (3.0*M/(4.0*np.pi*rhoM))**(1.0/3.0)

        return self.cosmology.sigmaR_of_z(R,z,tracer)

    def create_sigmaM_funcs(self):
        """
        This function creates the interpolating functions for sigmaM and dsigamM
        """
        M_inter = self.M
        z_inter = self.cosmology.results.zgrid

        sigmaM = self.sigmaM_of_z(M_inter,z_inter)

        # create interpolating functions
        logM_in_Msun = np.log(M_inter.to(u.Msun).value)
        logsigmaM = np.log(sigmaM)

        inter_logsigmaM = RectBivariateSpline(logM_in_Msun,z_inter,logsigmaM)

        #restore units
        def sigmaM_of_M_and_z(M,z):
            M = np.atleast_1d(M)
            z = np.atleast_1d(z)
            logM = np.log(M.to(u.Msun).value)
            return np.squeeze(np.exp(inter_logsigmaM(logM,z)))
        def dsigmaM_of_M_and_z(M,z):
            M = np.atleast_1d(M.to(u.Msun))
            sigmaM = np.reshape(sigmaM_of_M_and_z(M,z),(*M.shape,*z.shape))
            logM = np.log(M.value)
            return np.squeeze(sigmaM/M[:,None] * inter_logsigmaM.partial_derivative(1,0)(logM,z))


        return sigmaM_of_M_and_z, dsigmaM_of_M_and_z


    def mass_non_linear(self, z, delta_crit =1.686):
        '''
        Get (roughly) the mass corresponding to the nonlinear scale in units of Msun h
        '''
        sigmaM_z = self.sigmaM(self.M,z)
        mass_non_linear = self.M[np.argmin(np.power(sigmaM_z - delta_crit,2),axis=0)]

        return mass_non_linear.to(self.Msunh)

    def S3_dS3(self,M,z):
        '''
        The skewness and derivative with respect to mass of the skewness.
        Used to calculate the correction to the HMF due to non-zero fnl,
        as presented in 2009.01245.

        Their parameter k_cut is equivalent to our klim, not to be confused
        with the ncdm parameter. k_lim represents the cutoff in the skewness
        integral, we opt for no cutoff and thus set it to a very small value.
        This can be changed if necessary.
        '''
        tracer = self.astrotracer
        M = np.atleast_1d(M)
        z = np.atleast_1d(z)

        rhoM = self.rho_crit*self.cosmology.Omega(0,tracer=tracer)
        R = (3.0*M/(4.0*np.pi*rhoM))**(1.0/3.0)

        kmin= self.cosmology.results.kmin_pk
        kmax= self.cosmology.results.kmax_pk
        k = np.geomspace(kmin,kmax,128) / u.Mpc
        mu= np.linspace(-0.995,0.995,128)

        # Why Numpy is just the best

        #############################
        # Indicies k1, k2, mu, M, z #
        #############################

        # funnctions of k1 or k2 only
        P_phi = 9/25 * self.cosmology.primordial_scalar_pow(k)
        k_1 = k[:,None,None,None,None]
        k_2 = k[None,:,None,None,None]
        P_1 = P_phi[:,None,None,None,None]
        P_2 = P_phi[None,:,None,None,None]
        # functions of k1 or k2 and M
        tM = M[None,:]
        tR = R[None,:]
        tk = k[:,None]
        j1 = self.j1(tR*tk)
        dj1_dM = self.dj1(tR*tk) * (tk*tR)/(3 * tM)
        j1_1 = j1[:,None,None,:,None]
        j1_2 = j1[None,:,None,:,None]
        dj1_dM_1 = dj1_dM[:,None,None,:,None]
        dj1_dM_2 = dj1_dM[None,:,None,:,None]
        # functions of k1 or k2 and z
        Tm = -5/3 * np.reshape(self.cosmology.Transfer(k,z,nonlinear=False,tracer=tracer),(*k.shape,*z.shape))
        Tm_1 = Tm[:,None,None,None,:]
        Tm_2 = Tm[None,:,None,None,:]
        # functions of k1, k2, and mu
        tk_1 = k[:,None,None]
        tk_2 = k[None,:,None]
        tmu = mu[None,None,:]
        k_12 = np.sqrt(np.power(tk_1,2)+np.power(tk_2,2)+2*tk_1*tk_2*tmu)
        # functions of k1, k2, mu, and M
        tR = R[None,:]
        tk = k_12.flatten()[:,None]
        kmask = np.where((tk>kmin) and (tk<kmask))
        tx = tR * tk
        j1_12 = np.zeros_like(tx)
        dj1_dM_12 = np.zeros_like(tx)
        j1_12[kmask,:] = self.j1(tx[kmask,:])
        dj1_dM_12[kmask,:] = self.dj1(tx[kmask,:])*tx[kmask,:]/ (3 * tM)
        j1_12= np.reshape(j1_12,(len(k),len(k),len(mu),len(M)))[:,:,:,:,None]
        dj1_dM_12= np.reshape(dj1_dM_12,(len(k),len(k),len(mu),len(M)))[:,:,:,:,None]
        # functions of k1, k2, mu, and z
        Tm_12 = np.zeros((len(tk),len(z)))
        Tm_12[kmask,:] = -5/3 * np.reshape(self.cosmology.Transfer(tk[kmask],z,nonlinear=False,tracer=tracer),(*(tk[kmask].shape),*z.shape))
        Tm_12 = np.reshape(Tm_12,(*k.shape,*k.shape,*mu.shape,*z.shape))[:,:,:,None,:]

        # Integrandts
        W = j1_1 * j1_2 * j1_12
        dW_dM = j1_1 * j1_2 * dj1_dM_12 + j1_1 * dj1_dM_2 * j1_12 + dj1_dM_1 * j1_2 * j1_12

        integ_S3 =  np.power(k_1,2) * np.power(k_2,2)* W * Tm_1 * Tm_2 * Tm_12 * P_1 * P_2
        integ_dS3_dM = np.power(k_1,2) * np.power(k_2,2)* dW_dM * Tm_1 * Tm_2 * Tm_12 * P_1 * P_2

        # The integration
        S3 = np.trapz(integ_S3,k,axis=0)
        S3 = np.trapz(S3,k,axis=0)
        S3 = np.trapz(S3,mu,axis=0)
        dS3_dM = np.trapz(integ_dS3_dM,k,axis=0)
        dS3_dM = np.trapz(dS3_dM,k,axis=0)
        dS3_dM = np.trapz(dS3_dM,mu,axis=0)

        fac = self.cosmology.input_cosmoparams["f_NL"]*6/8/np.pi**4
        S3 = -1 * fac * np.squeeze(S3)
        dS3_dm = -1 * fac * np.squeeze(dS3_dm)
        return -S3, dS3_dm

    def kappa3_dkappa3(self,M,z):
        '''
        Calculates kappa_3 its derivative with respect to halo mass M from 2009.01245
        '''
        S3, dS3_dM = self.S3_dS3(M,z)
        sigmaM = self.sigmaM(M,z)
        dSigmaM = self.dsigmaM_dM(M,z)

        kappa3 = S3/sigmaM
        dkappa3dM = (dS3_dM - 3 * S3 * dSigmaM / sigmaM) /(sigmaM**3)

        return kappa3, dkappa3dM

    def Delta_HMF(self,M,z):
        '''
        The correction to the HMF due to non-zero f_NL, as presented in 2009.01245.
        '''
        sigmaM = self.sigmaM(M,z)
        dSigmaM = self.dsigmaM_dM(M,z)

        nuc = 1.42/sigmaM
        dnuc_dM = -1.42*dSigmaM/(sigmaM)**2

        kappa3, dkappa3_dM = self.kappa3_dkappa3(M,z)

        H2nuc = nuc**2-1
        H3nuc = nuc**3-3*nuc

        F1pF0p = (kappa3*H3nuc - H2nuc*dkappa3_dM/dnuc_dM )/6

        return F1pF0p

    def Delta_b(self,k,M,z):
        """
        Scale dependent correction to the halo bias in presence of primordial non-gaussianity
        """
        tracer = self.astrotracer

        M = np.atleast_1d(M)
        z = np.atleast_1d(z)
        k = np.atleast_1d(k)

        Tk = np.reshape(self.cosmology.Transfer(k,z),(*k.shape,*z.shape))
        sigmaM = self.sigmaM_of_z(M,z,tracer=tracer)
        bias_func = getattr(self.bias_function,self.astroparams["bias_model"])
        bias = np.reshape(bias_func(dc=self.delta_crit,nu=self.delta_crit/sigmaM),(*M.shape,*z.shape))

        f1 =  (self.cosmology.Hubble(0,physical=True) / (c.c  * k)).to(1).value
        f2 = 3*self.cosmology.Omega(0,tracer=tracer)*self.cosmology.input_cosmoparams["f_NL"]

        f1_of_k = f1[None,None,:]
        Tk_of_k_and_z = Tk[:,None,:]
        bias_of_M_and_z = bias[None,:,:]

        #Compute non-Gaussian correction Delta_b
        delta_b = (bias_of_M_and_z-1)*f2* f1_of_k/Tk_of_k_and_z
        return np.squeeze(delta_b)

    ##################
    # Helper Functions
    ##################
    # To be Outsorced
    def lognormal(self, x,mu,sigma):
        '''
        Returns a lognormal PDF as function of x with mu and sigma
        being the mean of log(x) and standard deviation of log(x), respectively
        '''
        try:
            return 1/x/sigma/(2.*np.pi)**0.5*np.exp(-(np.log(x.value) - mu)**2/2./sigma**2)
        except:
            return 1/x/sigma/(2.*np.pi)**0.5*np.exp(-(np.log(x) - mu)**2/2./sigma**2)

    def j1(self,x):
        W = 3 /np.power(x,3)*(np.sin(x*u.rad)-x*np.cos(x*u.rad))
        W[np.where(x<0.01)] = 1 - np.power(x[np.where(x<0.01)],2) / 10

        return W

    def dj1(self,x):
        dW = 3 /np.power(x,2)*np.sin(x*u.rad) -9 /np.power(x,4)*(np.sin(x*u.rad) - x*np.cos(x*u.rad))
        dW[np.where(x<0.01)] = -x/5 + np.power(x,3)/70

        return dW

    def set_astrophysics_defaults(self):
        """
        Fills up default values in the astropars dictionary if the values are not found.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        self.astroparams.setdefault("model_type", "LF")
        self.astroparams.setdefault("model_name", "SchCut")
        self.astroparams.setdefault(
            "model_par",
            {
                "phistar": 9.6e-11 * u.Lsun**-1 * u.Mpc**-3,
                "Lstar": 2.1e6 * u.Lsun,
                "alpha": -1.87,
                "Lmin": 5000 * u.Lsun,
            },
        )
        self.astroparams.setdefault("hmf_model", "ST")
        self.astroparams.setdefault("bias_model", "ST99")
        self.astroparams.setdefault("bias_par", {})
        self.astroparams.setdefault("nu", 115 * u.GHz)
        self.astroparams.setdefault("nuObs", 30 * u.GHz)
        self.astroparams.setdefault("Mmin", 1e9 * u.Msun)
        self.astroparams.setdefault("Mmax", 1e15 * u.Msun)
        self.astroparams.setdefault("nM", 500)
        self.astroparams.setdefault("Lmin", 10 * u.Lsun)
        self.astroparams.setdefault("Lmax", 1e8 * u.Lsun)
        self.astroparams.setdefault("nL", 5000)
        self.astroparams.setdefault("v_of_M", None)
        self.astroparams.setdefault("line_incli", True)
        self.astroparams.setdefault("astro_tracer","clustering")

    def init_model(self):
        '''
        Check if model given by model_name exists in the given model_type
        '''
        model_type = self.astroparams["model_type"]
        model_name = self.astroparams["model_name"]
        self.luminosity_function = lf.luminosity_functions(self)
        self.mass_luminosity_function = ml.mass_luminosity(self)

        if model_type=='ML' and not hasattr(self.mass_luminosity_function,model_name):
            if hasattr(self.luminosity_function,model_name):
                raise ValueError(model_name+" not found in mass_luminosity.py."+
                        " Set model_type='LF' to use "+model_name)
            else:
                raise ValueError(model_name+
                        " not found in mass_luminosity.py")

        elif model_type=='LF' and not hasattr(self.luminosity_function,model_name):
            if hasattr(self.mass_luminosity_function,model_name):
                raise ValueError(model_name+
                        " not found in luminosity_functions.py."+
                        " Set model_type='ML' to use "+model_name)
            else:
                raise ValueError(model_name+
                        " not found in luminosity_functions.py")

    def init_bias_function(self):
        '''
        Initialise computation of bias function if model given by bias_model exists in the given model_type
        '''
        bias_name = self.astroparams["bias_model"]
        self.bias_function = bf.bias_fittinig_functions(self)
        if not hasattr(self.bias_function,bias_name):
            raise ValueError(bias_name+
                        " not found in bias_fitting_functions.py")

    def init_halo_mass_function(self):
        '''
        Initialise computation of halo mass function if model given by hmf_model exists in the given model_type
        '''
        hmf_model = self.astroparams["hmf_model"]
        self.halo_mass_function = HMF.halo_mass_functions(self)

        if not hasattr(self.halo_mass_function,hmf_model):
            raise ValueError(hmf_model+
                        " not found in halo_mass_functions.py")

    def recap_astro(self):
        print("Astronomical Parameters:")
        for key in self.astroparams:
            print("   " + key + ": {}".format(self.astroparams[key]))
