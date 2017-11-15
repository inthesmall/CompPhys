import matrix
import numpy as np


A = np.matrix(
    '2 1 0 0 0 ; 3 8 4 0 0 ; 0 9 20 10 0 ; 0 0 22 51 -25 ; 0 0 0 -55 60')
b = np.array([2, 5, -4, 8, 9])
print("A")
print(A)
P, n, L, U = matrix.LU(A)
print("L")
print(L)
print("U")
print(U)
print("P")
print(P)
print("LU")
print(L * U)
print("PA")
print(P * A)
print("det(A)", matrix.LU_det(n, U))
print("x")
print(matrix.simul_solve(A, b))
print("A inverse")
print(matrix.invert(A))
