import matrix
import numpy as np
import scipy.linalg as scl
import unittest


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


class IntTest(unittest.TestCase):
    def setUp(self):
        self.A = np.matrix(
            '2 1 0 0 0 ; 3 8 4 0 0 ; 0 9 20 10 0 ; 0 0 22 51 -25 ; 0 0 0 -55 60')
        self.P, self.n, self.L, self.U = matrix.LU(self.A)

    def test_matrix_of_ints(self):
        self.assertTrue((self.L * self.U == self.P * self.A).all())


class Simulfbsub(unittest.TestCase):
    def setUp(self):
        self.M = np.matrix('1 2; 1 1')
        self.b = np.array([[3], [5]])
        self.P, self.n, self.L, self.U = matrix.LU(self.M)

    def test_simul_LU(self):
        self.assertTrue(((self.L * self.U) == (self.P * self.M)).all())

    def test_simul_P(self):
        self.assertTrue((self.P == np.matrix('0 1;1 0')).all())

    def test_simul_solve(self):
        self.assertTrue(
            (matrix.fb_sub(self.L, self.U, self.P, self.b) ==
                np.array([[7.], [-2.]])).all())


class GeneralSimulfbsub(unittest.TestCase):
    def test_general_simul_solve_fbsub(self):
        for i in range(1, 11):
            M = np.matrix(np.random.random(size=(i, i)))
            b = np.random.random(size=(i, 1))
            P, n, L, U = matrix.LU(M)
            x = matrix.fb_sub(L, U, P, b)
            x_true = np.matrix(scl.inv(M)) * b
            self.assertTrue(((x - x_true) < 1e-10).all())


class General_simulsolve(unittest.TestCase):
    def test_general_simul_solve(self):
        for i in range(1, 11):
            M = np.matrix(np.random.random(size=(i, i)))
            b = np.random.random(size=(i, 1))
            x = matrix.simul_solve(M, b)
            x_true = np.matrix(scl.inv(M)) * b
            self.assertTrue(((x - x_true) < 1e-10).all())
