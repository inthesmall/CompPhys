import numpy as np


def LU(M):
    """
    Decompose a square matrix M into upper and lower triangular matrices

    Implements Crout's method of LU decomposition with partial pivoting to
    decompose a matrix M such that L * U = P * M, where P is the permutation
    matrix. Also returns n as either -1 or 1 and is the determinant of P.

    Params:
        M: np.matrix of ints or floats. Square
            Matrix to be decomposed

    Returns:
        P: np.matrix of ints. Square
            Permutation matrix
        n: int. -1 or 1
            Determinant of P
        L: np.matrix of floats. Square
            Lower triangular matrix. Unity along the diagonal
        U: np.matrix of floats. Square
            Upper triangular matrix
    """
    M = M.copy()
    # P is permutation matrix. n is +/- 1 depending on number of swaps made
    P, n = _pivot(M)
    # We decompose a row-wise permutation of the original matrix
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
    # We decompose M in-place since this is faster and less memory-intensive
    # so we end up with M being a combined matrix, which we need to split
    # into L and U
    return P, n, _Lower(M), _Upper(M)


def _Lower(LU):
    # Find the lower matrix from the combined matrix
    n = len(LU)
    return np.reshape(np.matrix([((LU[i, j], 0)[i < j], 1)[i == j]
                                 for i in range(n) for j in range(n)]), (n, n))


def _Upper(LU):
    # Find the upper matrix from the combined matrix
    n = len(LU)
    return np.reshape(np.matrix([(LU[i, j], 0)[i > j]
                                 for i in range(n) for j in range(n)]), (n, n))


def LU_det(n, U):
    det = n
    for i in range(len(U)):
        det *= U[i, i]
    return det


def _pivot(M):
    # store used rows
    used = []
    # Permutations
    P = []
    for col in range(len(M)):
        if _check_unused(col, M, used, P):
            continue
        elif _repivot_used(col, M, used, P):
            continue
        else:
            raise ValueError("Col of 0's!")
    return _matrix_P(P), _det_P(P)


def _check_unused(col, M, used, P):
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


def _repivot_used(col, M, used, P):
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
        if _check_unused(used_col, M, used, P):
            try:
                P[col] = entry[1]
            except IndexError:
                P.append(entry[1])
            return True
    else:
        # See if we can use another used row to replace the one we want
        for entry in entries:
            used_col = P.index(entry[1])
            if _repivot_used(used_col, M, used, P):
                try:
                    P[col] = entry[1]
                except IndexError:
                    P.append(entry[1])
                return True
        else:
            return False


def _det_P(v):
    n = 1
    n *= (-1) ** v[0]

    for i, vi in enumerate(v[1:-1]):
        viold = vi
        for vj in v[:i + 1]:
            if vj < viold:
                vi -= 1
        n *= (-1) ** vi
    return n


def _matrix_P(P):
    # Create a permutation matrix from an ordered list of row numbers
    return np.matrix([[(0, 1)[i == P[n]] for i in range(len(P))]
                      for n in range(len(P))])


def fb_sub(L, U, P, b):
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
            column vector containing the right hand side vector of the
            simultaneous equation

    Returns:
        x: np.array of floats
            array containing the solutions to the simultaneous equations
    """
    # Cast input to floats
    L = L.astype(float)
    U = U.astype(float)
    b = b.astype(float)
    # attempt to make b into a column vector if provided as a row vector.
    try:
        if b.shape[1] != 1:
            b = b.reshape(len(b), 1)
    except IndexError:
        b = b.reshape(len(b), 1)
    # permute b to match L and U
    b = P * b
    # forward substitution for y:
    y = [b[0] / L[0, 0]]
    for i in range(1, len(b)):
        yi = b[i]
        for j in range(i):
            yi -= L[i, j] * y[j]
        yi /= L[i, i]
        y.append(yi)
    # back substitution for x:
    N = len(b) - 1
    x = np.zeros(N + 1, dtype='float')
    x[N] = y[N] / U[N, N]
    for i in range(N - 1, -1, -1):
        xi = y[i]
        for j in range(N, i, -1):
            xi -= U[i, j] * x[j]
        xi /= U[i, i]
        x[i] = xi
    return x
