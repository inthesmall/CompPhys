import numpy as np
import scipy.sparse as ss

import objects
import tools
from core import *


# step size in the x and y direction

# Conductivity of silicon, ceramic, alu in W/mm K

k = np.array([i.get_k() for i in objects.no_heatsink])
values = np.array([(1, 0)[isinstance(i, objects.Outside)]
                   for i in objects.no_heatsink])

up = np.array([(j, 0)[i % across == 0] for i, j in enumerate(values)])
down = np.array([(j, 0)[(i + 1) % (across) == 0]
                 for i, j in enumerate(values)])
down[0] = values[0]

D_inv = ss.dia_matrix((values * -0.25, [0]), shape=(total, total))
U = ss.dia_matrix(
    (np.stack((up, values)), [1, across]), shape=(total, total))
L = ss.dia_matrix(
    (np.stack((down, values)), [-1, -across]), shape=(total, total))


def update_edges(obj, u):
    across = obj.across
    return [i.update(u, j, across) for j, i in enumerate(obj.item)]

def run(target):
    u = np.full(total, 24000.)
    old = 0
    iters = 0
    update = D_inv * (U + L)

    while abs(sum(u) - old) > target:
        old = sum(u)
        edges = get_edges(u)
        u = -update * u - (D_inv * (edges + k))
        iters += 1
    return u, iters


def run_num(num):
    u = np.full(total, 20000.)
    update = D_inv * (U + L)
    for ki in range(num):
        edges = get_edges(u)
        u = -update * u - (D_inv * (edges + k))
    return u


def run_object(obj, target):
    item = obj.item
    total = obj.total
    across = obj.across

    k = np.array([i.get_k() for i in item])
    values = np.array(
        [0 if isinstance(i, objects.Outside) else 1 for i in item])

    up = np.array([(j, 0)[i % across == 0] for i, j in enumerate(values)])
    down = np.array([(j, 0)[(i + 1) % (across) == 0]
                     for i, j in enumerate(values)])
    down[0] = values[0]

    D_inv = ss.dia_matrix((values * -0.25, [0]), shape=(total, total))
    U = ss.dia_matrix(
        (np.stack((up, values)), [1, across]), shape=(total, total))
    L = ss.dia_matrix(
        (np.stack((down, values)), [-1, -across]), shape=(total, total))

    u = np.full(total, 4190.)
    old = 0
    iters = 0
    update = D_inv * (U + L)

    while abs(sum(u) - old) > target:
        old = sum(u)
        edges = get_edges(u, obj)
        u = -update * u - (D_inv * (edges + k))
        iters += 1
    return u, iters


def run_object_num(obj, num):
    item = obj.item
    total = obj.total
    across = obj.across

    k = np.array([i.get_k() for i in item])
    values = np.array(
        [0 if isinstance(i, objects.Outside) else 1 for i in item])

    up = np.array([(j, 0)[i % across == 0] for i, j in enumerate(values)])
    down = np.array([(j, 0)[(i + 1) % (across) == 0]
                     for i, j in enumerate(values)])
    down[0] = values[0]

    D_inv = ss.dia_matrix((values * -0.25, [0]), shape=(total, total))
    U = ss.dia_matrix(
        (np.stack((up, values)), [1, across]), shape=(total, total))
    L = ss.dia_matrix(
        (np.stack((down, values)), [-1, -across]), shape=(total, total))

    u = np.full(total, 400.)
    update = D_inv * (U + L)
    for ki in range(num):
        boundary_conds = update_edges(obj, u)
        u = -update * u - (D_inv * (boundary_conds + k))
    return u
