import numpy as np
import matrix

from functools import lru_cache as cache


# @cache()
def _spline(xs, ys):
    L = len(xs)
    M = np.matrix(np.zeros((L - 2, L - 2)), dtype=float)
    b = np.zeros((L - 2, 1), dtype=float)

    M[0, 0] = (xs[2] - xs[0]) / 3.
    M[0, 1] = (xs[2] - xs[1]) / 6.
    b[0, 0] = ((ys[2] - ys[1]) / (xs[2] - xs[1])) - \
              ((ys[1] - ys[0]) / (xs[1] - xs[0]))

    M[L - 3, L - 4] = (xs[L - 2] - xs[L - 3]) / 6.
    M[L - 3, L - 3] = (xs[L - 1] - xs[L - 3]) / 3.
    b[L - 3, 0] = ((ys[L - 1] - ys[L - 2]) / (xs[L - 1] - xs[L - 2])) - \
        ((ys[L - 2] - ys[L - 3]) / (xs[L - 2] - xs[L - 3]))

    for i in range(1, L - 3):
        M[i, i - 1] = (xs[i + 1] - xs[i]) / 6.
        M[i, i] = (xs[i + 2] - xs[i]) / 3.
        M[i, i + 1] = (xs[i + 2] - xs[i + 1]) / 6.
        b[i, 0] = ((ys[i + 2] - ys[i + 1]) / (xs[i + 2] - xs[i + 1])) - \
            ((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]))

    ypp = matrix.simul_solve(M, b)
    return ypp


def spline_inter(x, xs, ys):
    ypp = _spline(xs, ys)
    return y
