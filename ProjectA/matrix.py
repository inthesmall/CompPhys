import numpy as np


def LU(M):
    M = M.copy()
    P, n = pivot(M)
    M = P * M
    for j in range(len(M)):
        for i in range(j + 1):
            mij = M[i, j]
            if i != 0:
                for k in range(0, i):
                    mij -= M[i, k] * M[k, j]
            M[i, j] = mij
        for i in range(j + 1, len(M)):
            mij = M[i, j]
            for k in range(j):
                mij -= M[i, k] * M[k, j]
            mij /= M[j, j]
            M[i, j] = mij
    return P, n, Lower(M), Upper(M)


def Lower(LU):
    n = len(LU)
    return np.reshape(np.matrix([((LU[i, j], 0)[i < j], 1)[i == j]
                                 for i in range(n) for j in range(n)]), (n, n))


def Upper(LU):
    n = len(LU)
    return np.reshape(np.matrix([(LU[i, j], 0)[i > j]
                                 for i in range(n) for j in range(n)]), (n, n))


def LU_det(n, U):
    det = n
    for i in range(len(U)):
        det *= U[i, i]
    return det


def pivot(M):
    # store used rows
    used = []
    # Permutations
    P = []
    for col in range(len(M)):
        if check_unused(col, M, used, P):
            continue
        elif repivot_used(col, M, used, P):
            continue
        else:
            raise ValueError("Col of 0's!")
    return matrix_P(P), det_P(P)


def check_unused(col, M, used, P):
    entries = []
    # find any unused rows that have a non-zero entry for col
    for row in [i for i in range(len(M)) if i not in used and
                M[i, col] != 0]:
        entries.append((M[row, col], row))
    if not entries:
        # give up if there aren't any suitable
        return False
    entries.sort(reverse=True)
    # Choose the biggest entry and put it in the appropriate places
    used.append(entries[0][1])
    try:
        P[col] = entries[0][1]
    except IndexError:
        P.append(entries[0][1])
    return True


def repivot_used(col, M, used, P):
    entries = []
    # find all the rows in used with a non-zero entry for col
    # apart from the row which is already being used for col
    for row in used:
        if M[row, col] != 0 and P.index(row) != col:
            entries.append((M[row, col], row))
    entries.sort(reverse=True)
    for entry in entries:
        used_col = P.index(entry[1])
        # See if you can replace the used row with an unused one
        if check_unused(used_col, M, used, P):
            try:
                P[col] = entry[1]
            except IndexError:
                P.append(entry[1])
            return True
    else:
        # See if we can use another used row to replace the one we want
        for entry in entries:
            used_col = P.index(entry[1])
            if repivot_used(used_col, M, used, P):
                try:
                    P[col] = entry[1]
                except IndexError:
                    P.append(entry[1])
                return True
        else:
            return False


def det_P(v):
    n = 1
    n *= (-1)**v[0]

    for i, vi in enumerate(v[1:-1]):
        viold = vi
        for vj in v[:i + 1]:
            if vj < viold:
                vi -= 1
        n *= (-1)**vi
    return n


def matrix_P(P: list) -> np.matrix:
    """
    Create a permutation matrix from an ordered list of row numbers

    Params:
        P: list of row numbers of the identity matrix in the order they are to
            appear in the permutation matrix

    Returns:
        A row-wise permutation of the identity matrix as a numpy matrix
    """
    return np.matrix([[(0, 1)[i == P[n]] for i in range(len(P))]
                      for n in range(len(P))])


def fb_sub(L: np.matrix, U: np.matrix, P: np.array, b: np.array) -> np.array:
    """
    Performs forward- and back-substitution to solve simultaneous equations

    Solve a system of simultaneous equations using forward- and back-
    substitution given an upper- and lower-triangular matrix and a vector

    Designed to be called with output from LU(). Unless you have a good reason,
    it is better to use simul_solve() which does all the legwork for you.

    Params:
        L: np.matrix of ints or floats. Square
            Lower triangular matrix from LU decomposition. From LU(M)
        U: np.matrix of ints or floats. Square
            Upper triangular matrix from LU decomposition. From LU(M)
        P: np.matrix of ints. Square row-wise permutation of the identity
            Permutation matrix from LU decomposition. From LU(M) or pivot(M)
        b: np.array of ints or floats
            array containing the right hand side vector of the simultaneous
            equation

    Returns:
        x: np.array of floats
            array containing the solutions to the simultaneous equations
    """
    # forward substitution for y:
    y = [b[0] / L[0, 0]]
    for i in range(1, len(b)):
        yi = b[i]
        for j in range(i-1):
            yi -= L[i, j] * y[j]
        yi /=L[i, i]
        y.append(yi)

    # back substitution for x:
    N = len(b)
    x = np.zeros(N, dtype='float')
    x[N] = y[N] / U[N, N]
    for i in range(N-1, -1, -1):
        xi = y[i]
        for j in range(N, i, -1):
            xi -= U[i, j] * x[j]
        xi /= U[i, i]
        x[i] = xi
    return x
