import numpy as np
import scipy.sparse as ss
import scipy.linalg as spl
import scipy.sparse.linalg as ssl

import test
import objects
import new as new_

from core import *


def a(obj):
    item = obj.item
    total = obj.total
    across = obj.across

    k = np.array([i.get_k() for i in item])
    values = np.array(
        [0 if isinstance(i, objects.Outside) else 1 for i in item])

    up = np.array([0 if i % across == 0 else j for i, j in enumerate(values)])
    down = np.array([0 if (i + 1) % (across) == 0 else j
                     for i, j in enumerate(values)])
    down[0] = values[0]

    D_inv = ss.dia_matrix((values * -0.25, [0]), shape=(total, total))
    D = ss.dia_matrix(([4] * total, [0]), shape=(total, total))
    U = ss.dia_matrix(
        (np.stack((up, values)), [1, across]), shape=(total, total))
    L = ss.dia_matrix(
        (np.stack((down, values)), [-1, -across]), shape=(total, total))
    Lp = ss.dia_matrix(
        (np.stack((down, [1] * total)), [-1, -across]), shape=(total, total))
    return D_inv, D, U, L, Lp, k


def jacobi(obj):
    D_inv, U, L, k = test.a(obj)
    update = D_inv * (U + L)
    inv = D_inv
    return update, inv, k


def g_strauss(obj):
    D_inv, U, L, k = test.a(obj)
    inv = np.matrix(spl.inv(L + spl.inv(D_inv)))
    update = inv * U
    return update, inv, k


def run_target(obj, target, method=jacobi, u=None):
    update, inv, k = method(obj)
    total = len(inv)

    if u is None:
        u = np.full(total, 400.)

    old = 0
    new = sum(u)
    iters = 0

    while abs(new - old) > target:
        print(iters)
        print(new-old)
        old = new
        edges = test.update_edges(obj, u)
        u = -update * u.reshape(-1,1) - (inv * (edges + k).reshape(-1,1))
        iters += 1
        new = sum(u)

    return u, iters


def run_num(obj, num, method=jacobi, u=None):
    update, inv, k = method(obj)
    total = len(inv)


    if u is None:
        u = np.full(total, 400.)

    for ki in range(num):
        edges = test.update_edges(obj, u)
        u = -update * u.reshape(-1,1) - (inv * (edges + k).reshape(-1,1))

    return u


def inspect_difference(obj, target, method=jacobi, u=None):
    update, inv, k = method(obj)
    total = obj.total

    if u is None:
        u = np.full(total, 400.)

    old = 0
    new = sum(u)
    iters = 0
    results = []

    while abs(new - old) > target:
        results.append(abs(new - old))
        old = new
        edges = new_.update_edges(obj, u)
        u = -update * u - (inv * (edges + k))
        iters += 1
        new = sum(u)

    return u, iters, results
