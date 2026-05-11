from functools import partial

import numpy as np
from scipy.interpolate import UnivariateSpline as _UnivariateSpline

from SSLimPy.utils.utils import linear_interpolate

UnivariateSpline = partial(_UnivariateSpline, s=0)


# --- FFTLog Approximator class ---
class FFTLog:
    def __init__(self, xgrid, fgrid, xmin, xmax, logN=8):
        # Fill function with additional key word arguments
        self.xmin = xmin
        self.xmax = xmax
        self.N = 2**logN

        # Log grid
        self.x = np.geomspace(xmin, xmax, self.N)
        self.logx = np.log(self.x)
        self.dlogx = self.logx[1] - self.logx[0]
        self.nu = np.fft.fftfreq(self.N, d=self.dlogx)

        logf_unbiased = np.empty_like(self.x)
        inmask = np.logical_and(
            self.x > np.min(xgrid),
            self.x < np.max(xgrid),
        )

        logf_unbiased[inmask] = UnivariateSpline(np.log(xgrid), np.log(fgrid))(
            np.log(self.x)[inmask]
        )
        logf_unbiased[~inmask] = linear_interpolate(
            np.log(xgrid), np.log(fgrid), np.log(self.x)[~inmask]
        )

        dlogfdlogx = UnivariateSpline(self.logx, logf_unbiased).derivative(1)(self.logx)
        self.q = np.mean(dlogfdlogx)
        self.gamma = self.q + 1j * 2 * np.pi * self.nu

        # Apply bias and FFT
        fx = np.exp(logf_unbiased) * self.x**-self.q
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
