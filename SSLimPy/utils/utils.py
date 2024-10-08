from numba import njit, prange
import numpy as np

#####################
# Special Functions #
#####################


@njit
def legendre_0(mu):
    return np.ones_like(mu)


@njit
def legendre_2(mu):
    return 1 / 2 * (3 * np.power(mu, 2) - 1)


@njit
def legendre_4(mu):
    return 1 / 8 * (35 * np.power(mu, 4) - 30 * np.power(mu, 2) + 3)


##################
# Base Functions #
##################


@njit(
    "(float64, float64, float64, float64, float64, float64)",
    fastmath=True,
)
def scalarProduct(k1, mu1, ph1, k2, mu2, ph2):
    radicant = (1 - mu1**2) * (1 - mu2**2)
    if radicant < 0:
        radicant = 0
    return k1 * k2 * (np.sqrt(radicant) * np.cos(ph1 - ph2) + mu1 * mu2)


@njit(
    "(float64, float64, float64, float64, float64, float64)",
    fastmath=True,
)
def addVectors(
    k1,
    mu1,
    ph1,
    k2,
    mu2,
    ph2,
):
    k1pk2 = scalarProduct(k1, mu1, ph1, k2, mu2, ph2)
    radicant = k1**2 + 2 * k1pk2 + k2**2
    if np.isclose(radicant, 0):
        k12 = 0
        mu12 = 0
        phi12 = 0
    else:
        k12 = np.sqrt(radicant)
        mu12 = (k1 * mu1 + k2 * mu2) / k12
        if np.isclose(np.abs(mu12), 1):
            phi12 = 0
        else:
            phi12 = np.arctan2(
                k1 * np.sqrt(1 - mu1**2) * np.sin(ph1)
                + k2 * np.sqrt(1 - mu2**2) * np.sin(ph2),
                k1 * np.sqrt(1 - mu1**2) * np.cos(ph1)
                + k2 * np.sqrt(1 - mu2**2) * np.cos(ph2),
            )
    return k12, mu12, phi12


@njit(
    "(float64[::1], float64[::1], float64[::1])",
    fastmath=True,
)
def linear_interpolate(xi, yi, x):
    xl = yi.size
    rxl = x.size
    assert xl == xi.size, "xi should be the same size as yi"

    # Find the indices of the grid points surrounding xi
    # Handle linear extrapolation for larger x
    x1_idx = np.searchsorted(xi, x)
    x1_idx[np.where(x1_idx == 0)] = 1
    x1_idx[np.where(x1_idx == xl)] = xl - 1
    x2_idx = x1_idx - 1

    x1, x2 = xi[x1_idx], xi[x2_idx]

    results = np.empty(rxl)
    for i in range(rxl):
        y1 = yi[x1_idx[i]]
        y2 = yi[x2_idx[i]]

        results[i] = (y2 * (x1[i] - x[i]) + y1 * (x[i] - x2[i])) / (x1[i] - x2[i])
    return results


@njit(
    "(float64[::1], float64[::1], float64[:,:], float64[::1], float64[::1])",
    fastmath=True,
)
def bilinear_interpolate(xi, yj, zij, x, y):
    # Check input sizes
    xl, yl = zij.shape
    rxl = x.size
    ryl = y.size
    assert xl == xi.size, "xi should be the same size as axis 0 of zij"
    assert yl == yj.size, "yj should be the same size as axis 1 of zij"
    assert ryl == rxl, "for every x should be a y"

    # Find the indices of the grid points surrounding (xi, yi)
    # Handle linear extrapolation for larger x,y
    x1_idx = np.searchsorted(xi, x)
    x1_idx[np.where(x1_idx == 0)] = 1
    x1_idx[np.where(x1_idx == xl)] = xl - 1
    x2_idx = x1_idx - 1

    y1_idx = np.searchsorted(yj, y)
    y1_idx[np.where(y1_idx == 0)] = 1
    y1_idx[np.where(y1_idx == yl)] = yl - 1
    y2_idx = y1_idx - 1

    # Get the coordinates of the grid points
    x1, x2 = xi[x1_idx], xi[x2_idx]
    y1, y2 = yj[y1_idx], yj[y2_idx]

    results = np.empty(rxl)
    for i in range(rxl):
        # Get the values at the grid points
        Q11 = zij[x1_idx[i], y1_idx[i]]
        Q21 = zij[x2_idx[i], y1_idx[i]]
        Q12 = zij[x1_idx[i], y2_idx[i]]
        Q22 = zij[x2_idx[i], y2_idx[i]]

        results[i] = (
            Q11 * (x2[i] - x[i]) * (y2[i] - y[i])
            + Q21 * (x[i] - x1[i]) * (y2[i] - y[i])
            + Q12 * (x2[i] - x[i]) * (y[i] - y1[i])
            + Q22 * (x[i] - x1[i]) * (y[i] - y1[i])
        ) / ((x2[i] - x1[i]) * (y2[i] - y1[i]))
    return results


# The numba trapezoid for phi, muq, and q
@njit("(float64[::1], float64[::1])", fastmath=True)
def trapezoid(y, x):
    s = 0.0
    for i in range(x.size - 1):
        dx = x[i + 1] - x[i]
        dy = y[i] + y[i + 1]
        s += dx * dy
    return s * 0.5


#########################
# Specialized Functions #
#########################


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
                    muq_integrand[imuq] = trapezoid(phi_integrand, deltaphi)
                q_integrand[iq] = trapezoid(muq_integrand, muq)
            Pconv[ik, imu] = trapezoid(q_integrand * q, np.log(q))
    return Pconv


@njit(
    "(uint16, uint16, "
    + "float64[:,:], float64[:,:], float64[:,:], "
    + "float64[:,:], float64[:,:], "
    + "float64[:,:])",
    parallel=True,
)
def construct_gaussian_cov(nk, nz, C00, C20, C40, C22, C42, C44):
    cov = np.empty((nk, 3, 3, nz))
    for ki in prange(nk):
        for zi in range(nz):
            cov[ki, 0, 0, zi] = C00[ki, zi]
            cov[ki, 1, 0, zi] = C20[ki, zi]
            cov[ki, 2, 0, zi] = C40[ki, zi]
            cov[ki, 0, 1, zi] = C20[ki, zi]
            cov[ki, 0, 2, zi] = C40[ki, zi]
            cov[ki, 1, 1, zi] = C22[ki, zi]
            cov[ki, 1, 2, zi] = C42[ki, zi]
            cov[ki, 2, 1, zi] = C42[ki, zi]
            cov[ki, 2, 2, zi] = C44[ki, zi]
    return cov
