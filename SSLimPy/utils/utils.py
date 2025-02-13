from numba import njit
import numpy as np

##################
# Helper Functions
##################
def lognormal(x, mu, sigma):
    """
    Returns a lognormal PDF as function of x with mu and sigma
    being the mean of log(x) and standard deviation of log(x), respectively
    """
    try:
        return (
            1
            / x
            / sigma
            / (2.0 * np.pi) ** 0.5
            * np.exp(-((np.log(x.value) - mu) ** 2) / 2.0 / sigma**2)
        )
    except AttributeError:
        return (
            1
            / x
            / sigma
            / (2.0 * np.pi) ** 0.5
            * np.exp(-((np.log(x) - mu) ** 2) / 2.0 / sigma**2)
        )


def restore_shape(A, *args):
    """
    Extremely dangerous function to reshape squeezed arrays into arrays with boradcastable shapes
    This assumes that the output shape has lenghs corresponding to input 
    and is sqeezed in order of the input
    """
    A = np.atleast_1d(A)
    inputShape = A.shape
    targetShape = ()
    for arg in args:
        targetShape = (*targetShape, *np.atleast_1d(arg).shape)

    inputShape = np.array(inputShape)
    targetShape = np.array(targetShape)

    new_shape_A = []
    j = 0
    for i in range(len(targetShape)):
        if j < len(inputShape) and inputShape[j] == targetShape[i]:
            new_shape_A.append(inputShape[j])
            j += 1
        else:
            new_shape_A.append(1)

    A = A.reshape(new_shape_A)
    return A

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


@njit
def smooth_W(x):
    lx = len(x)
    W = np.empty(lx)
    for ix, xi in enumerate(x):
        if xi < 1e-3:
            W[ix] = 1.0 - 1.0 / 10.0 * xi**2
        else:
            W[ix] = 3.0 / xi**3.0 * (np.sin(xi) - xi * np.cos(xi))
    return W


@njit
def smooth_dW(x):
    lx = len(x)
    dW = np.empty(lx)
    for ix, xi in enumerate(x):
        if xi < 1e-3:
            dW[ix] = -1.0 / 5.0 * xi - 1.0 / 70.0 * xi**3
        else:
            dW[ix] = 3.0 / xi**2.0 * np.sin(xi) - 9 / xi**4 * (
                np.sin(xi) - xi * np.cos(xi)
            )
    return dW


##################
# Base Functions #
##################


@njit(
    "(float64, float64, float64, float64, float64, float64)",
    fastmath=True,
)
def scalarProduct(k1, mu1, ph1, k2, mu2, ph2):

    s1s = 1 - mu1**2
    if np.isclose(s1s, 0):
        s1s = 0
        mu1 = 1 * np.sign(mu1)

    s2s = 1 - mu2**2
    if np.isclose(s2s, 0):
        s2s = 0
        mu2 = 1 * np.sign(mu2)

    return k1 * k2 * (np.sqrt(s2s * s1s) * np.cos(ph1 - ph2) + mu1 * mu2)


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
            # Yes this sometimes fails
            s1s = 1 - mu1**2
            if np.isclose(s1s, 0):
                s1s = 0
            s2s = 1 - mu2**2
            if np.isclose(s2s, 0):
                s2s = 0

            x = k1 * np.sqrt(s1s) * np.cos(ph1) + k2 * np.sqrt(s2s) * np.cos(ph2)
            y = k1 * np.sqrt(s1s) * np.sin(ph1) + k2 * np.sqrt(s2s) * np.sin(ph2)

            phi12 = np.arctan2(y, x)

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
