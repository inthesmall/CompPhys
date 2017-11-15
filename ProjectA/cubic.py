import numpy as np
import matrix

from functools import lru_cache as cache


# Use caching. This simplifies the interface. We can just keep calling
# function, and it will only perform the calculations once.
@cache()
def _spline(xs, ys):
    L = len(xs)
    # initialize the outputs
    M = np.matrix(np.zeros((L - 2, L - 2)), dtype=float)
    b = np.zeros((L - 2, 1), dtype=float)

    # Set up simultaneous equations
    # Treat the beginning and end separately
    M[0, 0] = (xs[2] - xs[0]) / 3.
    M[0, 1] = (xs[2] - xs[1]) / 6.
    b[0, 0] = ((ys[2] - ys[1]) / (xs[2] - xs[1])) - \
              ((ys[1] - ys[0]) / (xs[1] - xs[0]))

    M[L - 3, L - 4] = (xs[L - 2] - xs[L - 3]) / 6.
    M[L - 3, L - 3] = (xs[L - 1] - xs[L - 3]) / 3.
    b[L - 3, 0] = ((ys[L - 1] - ys[L - 2]) / (xs[L - 1] - xs[L - 2])) - \
        ((ys[L - 2] - ys[L - 3]) / (xs[L - 2] - xs[L - 3]))

    # Set the middle of the matrix
    for i in range(1, L - 3):
        M[i, i - 1] = (xs[i + 1] - xs[i]) / 6.
        M[i, i] = (xs[i + 2] - xs[i]) / 3.
        M[i, i + 1] = (xs[i + 2] - xs[i + 1]) / 6.
        b[i, 0] = ((ys[i + 2] - ys[i + 1]) / (xs[i + 2] - xs[i + 1])) - \
            ((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]))

    # Solve the matrix and return the second derivatives
    ypp = matrix.simul_solve(M, b)
    return ypp


# These functions give the A and C coefficients from the lecture notes
def _A(x, xs, i):
    return (xs[i + 1] - x) / (xs[i + 1] - xs[i])


def _C(x, xs, i, A):
    return ((A**3 - A) * (xs[i + 1] - xs[i])**2) / 6.


def spline_inter(x, xs, ys):
    """Perform cubic spline interpolation.

    Works like a function of *x*. Gives the interpolated value of the
    function at *x*.

    Params:
        x: float
            x value at which to evaluate interpolation
        xs: list
            x values of points in dataset
        ys: list
            y values of points in dataset

    Returns:
        y: float
            y value of function interpolated at *x*
    """
    # Trivial case, just give the matching y value
    if x in xs:
        return ys[xs.index(x)]

    # Calculate the second derivatives
    ypp = _spline(tuple(xs), tuple(ys))

    # Find where *x* lies in xs
    # Raise an error if the value can't be interpolated
    try:
        i = [i >= x for i in xs].index(True) - 1
    except ValueError as err:
        if err.args[0] == "True is not in list":
            raise ValueError("x is not in range of data") from err
        else:
            raise err
    if i == -1:
        raise ValueError("x is not in range of data")

    # Find the coefficients
    A = _A(x, tuple(xs), i)
    B = 1 - A
    C = _C(x, tuple(xs), i, A)
    D = _C(x, tuple(xs), i, B)

    # Calculate value and return
    if i == 0:
        return (A * ys[0] + B * ys[1] + D * ypp[0])[0]
    elif i == len(xs) - 2:
        return (A * ys[-2] + B * ys[-1] + C * ypp[-1])[0]
    else:
        return (A * ys[i] + B * ys[i + 1] + C * ypp[i - 1] + D * ypp[i])[0]
