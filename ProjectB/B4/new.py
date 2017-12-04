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
# values = np.array([1 for i in objects.no_heatsink])

up = np.array([(j, 0)[i % across == 0] for i, j in enumerate(values)])
down = np.array([(j, 0)[(i + 1) % (across) == 0]
                 for i, j in enumerate(values)])
down[0] = values[0]

D_inv = ss.dia_matrix((values * -0.25, [0]), shape=(total, total))
U = ss.dia_matrix(
    (np.stack((up, values)), [1, across]), shape=(total, total))
L = ss.dia_matrix(
    (np.stack((down, values)), [-1, -across]), shape=(total, total))




def get_edges(u):
    out = []
    for i, j in enumerate(objects.no_heatsink):
        if isinstance(j, objects.OutsideEdge):
            if isinstance(j, objects.Corner):
                out.append(2 * (u[i] - delta * 1.31E-6 *
                                (u[i] - 293.15)**(4 / 3)))
            elif isinstance(j, objects.InsideCorner):
                out.append(0)
            else:
                out.append(u[i] - delta * 1.31E-6 * (u[i] - 293.15)**(4 / 3))
        else:
            out.append(0)
    return np.array(out)
    return np.array([i.update(u, j) if isinstance(i, objects.OutsideEdge) else 0
                     for j, i in enumerate(objects.no_heatsink)])


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
