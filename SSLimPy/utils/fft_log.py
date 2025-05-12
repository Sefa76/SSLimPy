import numpy as np
from functools import partial

# --- FFTLog Approximator class ---
class FFTLog:
    def __init__(self, f, xmin, xmax, logN=8, q=0.0, **kwags):

        # Fill function with additional key word arguments
        pf = partial(f, **kwags)
        self.f = pf

        self.xmin = xmin
        self.xmax = xmax
        self.N = 2 ** logN
        self.q = q

        # Log grid
        self.x = np.geomspace(xmin, xmax, self.N)
        self.logx = np.log(self.x)
        self.dlogx = self.logx[1] - self.logx[0]
        self.nu = np.fft.fftfreq(self.N, d=self.dlogx)
        self.gamma = -self.q + 1j * 2 * np.pi * self.nu

        # Apply bias and FFT
        fx = self.f(self.x) * self.x**self.q
        self.C = np.fft.fft(fx) / self.N * self.xmin**(-1j * 2 * np.pi * self.nu)

        # Sort frequencies
        sort_idx = np.argsort(self.nu)
        self.gamma = self.gamma[sort_idx]
        self.C = self.C[sort_idx]
        self.C[0] *= 0.5
        self.C[-1] *= 0.5

    def get_power_and_coef(self):
        return self.gamma, self.C

    def __call__(self, x_eval):
        x_eval = np.atleast_1d(x_eval)
        approx = np.zeros_like(x_eval, dtype=complex)
        for i, gammai in enumerate(self.gamma):
            approx += self.C[i] * x_eval**(gammai)
        return np.real_if_close(approx)