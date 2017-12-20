import unittest
import main
import numpy as np


class TestTrapeze(unittest.TestCase):
    def test_sin_integration(self):
        res = main.trapeze(lambda x: np.sin(x), 0, np.pi, 1e-6)
        self.assertLess(abs(2 - res), 1e-5)


class TestSimpson(unittest.TestCase):
    def test_sin_integration(self):
        res = main.simpson(lambda x: np.sin(x), 0, np.pi, 1e-6)
        self.assertLess(abs(2 - res), 1e-5)


class TestMonte(unittest.TestCase):
    def test_sin_integration_flat(self):
        res = main.monte(lambda x: np.sin(x), 0, np.pi, 1e-6)
        self.assertLess(abs(res - 2), 0.005)

    def test_sin_integration_scaled(self):
        def rand(n=1):
            return -(-0.537 + np.sqrt(0.537**2 - 2 * 0.139 *
                                      np.random.random(n))) / 0.139

        def dist(x):
            return 0.537 - 0.139 * x

        res = main.monte(lambda x: np.sin(x), 0, np.pi, 1e-6, rand, dist)
        self.assertLess(abs(res - 2), 0.005)

    def test_sin_integration_scaled_generated_pdf(self):
        def dist(x):
            return 0.537 - 0.139 * x

        res = main.monte(np.sin, 0, np.pi, 1e-6, dist=dist)
        self.assertLess(abs(res - 2), 0.005)

    def test_dist(self):
        self.assertEqual(main.dist(1), 0.5)
