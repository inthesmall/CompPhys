import cubic
import numpy as np
import unittest


class splineTest(unittest.TestCase):
    def test_specific_case(self):
        xs = [0, 1, 2, 3, 4, 5]
        ys = [y**3 for y in xs]

        ypp = cubic._spline(xs, ys)

        out = [[5.85645933],
               [12.57416268],
               [15.84688995],
               [32.03827751]]

        diff = [y - o for (y, o) in zip(ypp, out)]
        sum_ = abs(sum(diff)[0])
        self.assertLess(sum_, 1e-6)
