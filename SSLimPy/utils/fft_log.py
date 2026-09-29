from functools import partial

import numpy as np
from scipy.interpolate import UnivariateSpline as _UnivariateSpline

UnivariateSpline = partial(_UnivariateSpline, s=0)

# --- FFTLog Approximator class ---
class FFTLog:
    def __init__(self, f, xmin, xmax, logN=8, args={}):
        pf = partial(f, **args)

        # Fill function with additional key word arguments

        self.N = 2**logN

        try:
            self.xmin = xmin.to(xmax.unit).value
            self.xmax = xmax.value
            self.x = np.geomspace(self.xmin, self.xmax, self.N) * xmax.unit
            self.logx = np.log(self.x.value)
        except AttributeError:
            self.xmin = xmin
            self.xmax = xmax
            self.x = np.geomspace(self.xmin, self.xmax, self.N)
            self.logx = np.log(self.x)

        self.dlogx = self.logx[1] - self.logx[0]
        self.nu = np.fft.fftfreq(self.N, d=self.dlogx)

        f_unbiassed = pf(self.x)
        try:
            f_unbiassed = f_unbiassed.value
            logf_unbiased = np.log(f_unbiassed.value)
        except AttributeError:
            logf_unbiased = np.log(f_unbiassed)

        dlogfdlogx = UnivariateSpline(self.logx, logf_unbiased).derivative(1)(self.logx)
        self.q = np.mean(dlogfdlogx)
        self.gamma = self.q + 1j * 2 * np.pi * self.nu

        # Apply bias and FFT
        fx = f_unbiassed * np.exp(-self.q * self.logx)
        self.C = np.fft.fft(fx) / self.N * self.xmin ** (-1j * 2 * np.pi * self.nu)

        # Sort frequencies
        sort_idx = np.argsort(self.nu)
        self.gamma = self.gamma[sort_idx]
        self.C = self.C[sort_idx]
        self.C[0] *= 0.5
        self.C[-1] *= 0.5

    def get_power_and_coef(self):
        return self.gamma, self.C

    def __call__(self, x_eval):
        x_eval = np.asarray(np.atleast_1d(x_eval), dtype=complex)
        approx = np.zeros_like(x_eval, dtype=complex)
        for i, gammai in enumerate(self.gamma):
            approx += self.C[i] * x_eval ** (gammai)
        return np.real_if_close(approx)
