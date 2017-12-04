import numpy as np

from core import *


class Point():
    def __init__(self):
        self.k = 0
        self.q = 0

    def get_k(self):
        if self.k == 0:
            return 0
        return (self.q * delta ** 2) / self.k

    def __float__(self):
        return float(self.k)


class OutsideEdge(Point):
    def __init__(self, k, q):
        self.k = k
        self.q = q / 2


# class LeftEdge(OutsideEdge):
#     def __init__(self, k, q):
#         super().__init__(k, q)

#     def update(self, u, j):
#         return u[j + 1] - (2 * delta * 1.31E-6 * (u[j] - 293.15)**(4 / 3))


# class RightEdge(OutsideEdge):
#     def __init__(self, k, q):
#         super().__init__(k, q)

#     def update(self, u, j):
#         return u[j - 1] + 2 * delta * 1.31E-6 * (u[j] - 293.15)**(4 / 3)


# class TopEdge(OutsideEdge):
#     def __init__(self, k, q):
#         super().__init__(k, q)

#     def update(self, u, j):
#         return u[j + across] - 2 * delta * 1.31E-6 * (u[j] - 293.15)**(4 / 3)


# class BottomEdge(OutsideEdge):
#     def __init__(self, k, q):
#         super().__init__(k, q)

#     def update(self, u, j):
#         return u[j - across] + 2 * delta * 1.31E-6 * (u[j] - 293.15)**(4 / 3)


class LeftEdge(OutsideEdge):
    def __init__(self, k, q):
        super().__init__(k, q)

    def update(self, u, j):
        return u[j] - (delta * 1.31E-6 * (u[j] - 293.15)**(4 / 3))


class RightEdge(OutsideEdge):
    def __init__(self, k, q):
        super().__init__(k, q)

    def update(self, u, j):
        return u[j] - delta * 1.31E-6 * (u[j] - 293.15)**(4 / 3)


class TopEdge(OutsideEdge):
    def __init__(self, k, q):
        super().__init__(k, q)

    def update(self, u, j):
        return u[j] - delta * 1.31E-6 * (u[j] - 293.15)**(4 / 3)


class BottomEdge(OutsideEdge):
    def __init__(self, k, q):
        super().__init__(k, q)

    def update(self, u, j):
        return u[j] - delta * 1.31E-6 * (u[j] - 293.15)**(4 / 3)

class Corner(OutsideEdge):
    def __init__(self, k, q):
        self.k = k
        self.q = q / 4


class TopLeftCorner(Corner):
    def __init__(self, k, q):
        super().__init__(k, q)

    def update(self, u, j):
        return (TopEdge.update(self, u, j) + LeftEdge.update(self, u, j))


class TopRightCorner(Corner):
    def __init__(self, k, q):
        super().__init__(k, q)

    def update(self, u, j):
        return (TopEdge.update(self, u, j) + RightEdge.update(self, u, j))


class BottomLeftCorner(Corner):
    def __init__(self, k, q):
        super().__init__(k, q)

    def update(self, u, j):
        return (BottomEdge.update(self, u, j) + LeftEdge.update(self, u, j))


class BottomRightCorner(Corner):
    def __init__(self, k, q):
        super().__init__(k, q)

    def update(self, u, j):
        return (BottomEdge.update(self, u, j) + RightEdge.update(self, u, j))


class Boundary(Point):
    def __init__(self, k1, k2, q):
        self.k = (k1 + k2) / 2
        self.q = q / 2


class Outside(Point):
    def __init__(self):
        super().__init__()


class Inside(Point):
    def __init__(self, k, q):
        self.k = k
        self.q = q


class InsideCorner(OutsideEdge):
    def __init__(self, k, q):
        self.k = k
        self.q = q

    def update(self, u, j):
        return 0


no_heatsink = np.concatenate((
    [TopLeftCorner(k2, 0)],
    np.full(across - 2, TopEdge(k2, 0)),
    [TopRightCorner(k2, 0)],
    np.tile(
        np.concatenate((
            [LeftEdge(k2, 0)],
            np.full(across - 2, Inside(k2, 0)),
            [RightEdge(k2, 0)])),
        int((2 / delta) - 1)
    ),
    [BottomLeftCorner(k2, 0)],
    np.full(int((3 / delta) - 1), BottomEdge(k2, 0)),
    [InsideCorner((2 * k2 + k1) / 3, q / 4)],
    np.full(int((14 / delta) - 1), Boundary(k1, k2, q)),
    [InsideCorner((2 * k2 + k1) / 3, q / 4)],
    np.full(int((3 / delta) - 1), BottomEdge(k2, 0)),
    [BottomRightCorner(k2, 0)],
    np.tile(
        np.concatenate((
            np.full(int(3 / delta), Outside()),
            [LeftEdge(k1, q)],
            np.full(int((14 / delta) - 1), Inside(k1, q)),
            [RightEdge(k1, q)],
            np.full(int(3 / delta), Outside()))),
        int((1 / delta) - 1)
    ),
    np.full(int(3 / delta), Outside()),
    [BottomLeftCorner(k1, q)],
    np.full(int((14 / delta) - 1), BottomEdge(k1, q)),
    [BottomRightCorner(k1, q)],
    np.full(int(3 / delta), Outside()),
))
