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


def pad_to_width(item, item_width, width):
    shape = np.reshape(item, (-1, int(item_width / delta) + 1))
    height = shape.shape[0]
    side_pad = (width - item_width) / (2 * delta)
    side = np.full((height, int(side_pad)), Outside())
    out = np.concatenate((side, shape, side), 1)
    return out.reshape(-1)


def make_heatsink_a(fins):
    width = (3 * fins) - 2
    heatsink = np.concatenate((
        # Top
        np.tile(
            np.concatenate((
                [TopLeftCorner(k3, 0)],
                np.tile(TopEdge(k3, 0), int(1 / delta) - 1),
                [TopRightCorner(k3, 0)],
                np.tile(Outside(), int(2 / delta) - 1))),
            fins - 1
        ),
        [TopLeftCorner(k3, 0)],
        np.tile(TopEdge(k3, 0), int(1 / delta) - 1),
        [TopRightCorner(k3, 0)],

        # Middle
        np.tile(
            np.concatenate((
                np.tile(
                    np.concatenate((
                        [LeftEdge(k3, 0)],
                        np.tile(Inside(k3, 0), int(1 / delta) - 1),
                        [RightEdge(k3, 0)],
                        np.tile(Outside(), int(2 / delta) - 1))),
                    fins - 1
                ),
                [LeftEdge(k3, 0)],
                np.tile(Inside(k3, 0), int(1 / delta) - 1),
                [RightEdge(k3, 0)]
            )),
            int(30 / delta) - 1
        ),

        # Base
        [LeftEdge(k3, 0)],
        np.tile(
            np.concatenate((
                np.tile(Inside(k3, 0), int(1 / delta) - 1),
                [InsideCorner(k3, 0)],
                np.tile(TopEdge(k3, 0), int(2 / delta) - 1),
                [InsideCorner(k3, 0)]
            )),
            fins - 1
        ),
        np.tile(Inside(k3, 0), int(1 / delta) - 1),
        [RightEdge(k3, 0)],
        # Ceramic
        np.tile(
            np.concatenate((
                [LeftEdge(k3, 0)],
                np.tile(Inside(k3, 0), int(width / delta) - 1),
                [RightEdge(k3, 0)]
            )),
            int(4 / delta) - 1
        )
    ))

    if width > 20:
        # pad cpu
        h = heatsink
        j = np.concatenate((
            [BottomLeftCorner(k3, 0)],
            np.tile(BottomEdge(k3, 0), int((width - 20) / (2 * delta)) - 1),
            [InsideCorner((2 * k3 + k2) / 3, 0)],
            np.tile(Boundary(k3, k2, 0), across - 2),
            [InsideCorner((2 * k3 + k2) / 3, 0)],
            np.tile(BottomEdge(k3, 0), int((width - 20) / (2 * delta)) - 1),
            [BottomRightCorner(k3, 0)]
        ))
        c = pad_to_width(no_heatsink[int(20 / delta) + 1:], 20, width)
        return np.concatenate((h, j, c))
    elif width < 20:
        # pad heatsink
        h = pad_to_width(heatsink, width, 20)
        j = np.concatenate((
            [TopLeftCorner(k2, 0)],
            np.tile(TopEdge(k2, 0), int((20 - width) / (2 * delta)) - 1),
            [InsideCorner((k3 + 2 * k2) / 3, 0)],
            np.tile(Boundary(k3, k2, 0), int(width / delta) - 1),
            [InsideCorner((k3 + 2 * k2) / 3, 0)],
            np.tile(TopEdge(k2, 0), int((20 - width) / (2 * delta)) - 1),
            [TopRightCorner(k2, 0)]
        ))
        c = no_heatsink[int(20 / delta) + 1:]
        return np.concatenate((h, j, c))
    else:
        out = np.append(no_heatsink, heatsink)

    return out


class HeatsinkA():
    def __init__(self, fins, delta=delta):
        self.item = make_heatsink_a(fins)
        self.total = len(self.item)
        self.width = max(20, (3 * fins) - 2)
        self.height = 37
        self.across = (self.width / delta) + 1
        self.down = (self.height / delta) + 1

    def __iter__(self):
        return self.item.__iter__()
