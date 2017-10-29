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
