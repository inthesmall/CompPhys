import new
import numpy as np
import objects
import scipy.sparse as ss
from core import *

u = np.full(3375, 500)


def speed_edges(obj):
    u = np.full(len(obj.item), 500)
    across = obj.across
    a = np.array([[u[j + across], u[j]]
                  if isinstance(i, objects.TopEdge) else [0, 0] for j, i in enumerate(obj.item)])
    a = a.T
    return a[0] + 2 * delta * 1.31E-6 * (a[1] - 293.15)**(4 / 3)


def find_edges(obj):
    a = np.array([[j + across, j]
                  if isinstance(i, objects.TopEdge) else [0, 0] for j, i
                  in enumerate(obj.item)])
    truth = (a.T[0] + a.T[1]) == 0
    return a, truth


def update(a, u, truth):
    at = a.T
    # b = u[at[0]] + 2 * delta * 1.31E-6 * (u[at[1]] - 293.15)**(4 / 3)
    return np.choose(truth, [u[at[0]] + 2 * delta * 1.31E-6 * (u[at[1]] - 293.15)**(4 / 3), 0])


def t(at, u):
    return u[at[0]] + 2 * delta * 1.31E-6 * (u[at[1]] - 293.15)**(4 / 3)


def t1(at, u, truth):
    a0 = np.choose(truth, [u[at[0]], 0])
    a1 = np.choose(truth, [u[at[1]], 0])
    return a0 + 2 * delta * 1.31E-6 * (a1 - 293.15)**(4 / 3)


def expand(u, mask):
    out = np.zeros(len(mask))
    i_u = 0
    for i, j in enumerate(mask):
        if j:
            out[i] = u[i_u]
            i_u += 1

    return out


def exp2(u, mask):
    u = list(u.copy())
    return np.array([u.pop() if i else 0 for i in mask[::-1]])

def update_edges(obj, u):
    across = obj.across
    mask = obj.mask
    u = expand(u, mask)
    return [i.update(u, j, across) for j, i in enumerate(obj.item)
            if not isinstance(i, objects.Outside)]


def a(obj):
    item = obj.item
    total = obj.total
    across = obj.across

    k = np.array([i.get_k() for i in item])
    values = obj.mask

    up = np.array([0 if i % across == 0 else j for i, j in enumerate(values)])
    down = np.array([0 if (i + 1) % (across) == 0 else j
                     for i, j in enumerate(values)])
    down[0] = values[0]

    D_inv = ss.dia_matrix(
        (values * -0.25, [0]), shape=(total, total)).toarray()
    U = ss.dia_matrix(
        (np.stack((up, values)), [1, across]), shape=(total, total)).toarray()
    L = ss.dia_matrix(
        (np.stack((down, values)), [-1, -across]), shape=(total, total)).toarray()

    mask = values.astype(bool)
    D_inv = np.matrix((D_inv[:, mask])[mask, :])
    U = np.matrix((U[:, mask])[mask, :])
    L = np.matrix((L[:, mask])[mask, :])
    k = k[mask]
    return D_inv, U, L, k
# def get_edges(u, obj=objects.no_heatsink):
#     try:
#         across = obj.across
#     except AttributeError:
#         print(1)
#     return np.array([i.update(u, j, across) if isinstance(i, objects.OutsideEdge) else 0
#                      for j, i in enumerate(obj)])
# def find_edges(obj):
#     for j,i in enumerate(obj.item):
#         if isinstance(i, ibjects.TopEdge):

#         elif isinstance(i, objects.BottomEdge):

#         elif
#     bounds = np.array([[j + across, j]
#                        if isinstance(i, objects.TopEdge) else [0, 0] for j, i
#                        in enumerate(obj.item)])
#     edges = (bounds.T[0] + bounds.T[1]) == 0
#     return bounds, edges


# def update_edges(bounds_t, u, edges):
#     # b = u[at[0]] + 2 * delta * 1.31E-6 * (u[at[1]] - 293.15)**(4 / 3)
#     return np.choose(edges, [u[bounds_t[0]] - 2 * delta * 1.31E-6 *
#                              (u[bounds_t[1]] - 293.15)**(4 / 3), 0])
