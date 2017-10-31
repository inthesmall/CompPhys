import unittest
import numpy as np
import matrix


class EdgeCase(unittest.TestCase):
    def setUp(self):
        self.M = np.matrix('2 1 0 0; 0 2 1 0 ; 0 0 2 1; 1 0 0 0')
        self.P, self.n, self.L, self.U = matrix.LU(self.M)

    def test_edge_lower(self):
        self.assertTrue((self.L == np.matrix(
            '1 0 0 0 ; 2 1 0 0 ; 0 2 1 0 ; 0 0 2 1')).all())

    def test_edge_upper(self):
        self.assertTrue((self.U == np.matrix(
            '1 0 0 0 ; 0 1 0 0 ; 0 0 1 0 ; 0 0 0 1')).all())

    def test_edge_det(self):
        self.assertEqual(-1, self.n)


class Simul(unittest.TestCase):
    def setUp(self):
        self.M = np.matrix('1 2; 1 1')
        self.b = np.array([[3], [5]])
        self.P, self.n, self.L, self.U = matrix.LU(self.M)

    def test_simul_LU(self):
        self.assertFalse(((self.L * self.U) - (self.P * self.M)).any())

    def test_simul_P(self):
        self.assertFalse((self.P - np.matrix('0 1;1 0')).any())

    def test_simul_solve(self):
        self.assertFalse(
            (matrix.fb_sub(self.L, self.U, self.P, self.b) -
                np.array([7., -2.])).any())
