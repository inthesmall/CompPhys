import cubic
import numpy as np
import unittest


class splineTest(unittest.TestCase):
    def test_specific_case(self):
        xs = [0, 1, 2, 3, 4, 5]
        ys = [y**3 for y in xs]

        ypp = cubic._spline(tuple(xs), tuple(ys))

        out = [[5.85645933],
               [12.57416268],
               [15.84688995],
               [32.03827751]]

        diff = [y - o for (y, o) in zip(ypp, out)]
        sum_ = abs(sum(diff)[0])
        self.assertLess(sum_, 1e-6)

    def test_with_cubic(self):
        xs = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        ys = [x**3 for x in xs]
        x_test = np.arange(-4, 4, 0.1)
        y_test = [cubic.spline_inter(x, xs, ys) for x in x_test]
        diff = [y - x**3 for (y, x) in zip(y_test, x_test)]
        sum_ = abs(sum(diff))
        self.assertLess(sum_, 1e-6)
