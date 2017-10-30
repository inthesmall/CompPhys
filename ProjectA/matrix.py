import numpy as np


def LU(M):
    # Not sure if this is Crout's method or Doolittle's
    M = M.copy()
    # P is permutation matrix. n is +/- 1 depending on number of swaps made
    P, n = pivot(M)
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
    # We end up with M being a combined matrix, which we need to split
    # into L and U
    return P, n, Lower(M), Upper(M)


def Lower(LU):
    n = len(LU)
    return np.reshape(np.matrix([((LU[i, j], 0)[i < j], 1)[i == j]
                                 for i in range(n) for j in range(n)]), (n, n))


def Upper(LU):
    n = len(LU)
    return np.reshape(np.matrix([(LU[i, j], 0)[i > j]
                                 for i in range(n) for j in range(n)]), (n, n))


def LUdet(n, U):
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
    return matrixP(P), detP(P)


def check_unused(col, M, used, P):
    entries = []
    # find any unused rows that have a non-zero entry for col
    for row in [i for i in range(len(M)) if i not in used and
                M[i, col] != 0]:
        entries.append((M[row, col], row))
    if entries == []:
        # give up if there arent't any suitable
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


def detP(v):
    n = 1
    n *= (-1)**v[0]

    for i, vi in enumerate(v[1:-1]):
        viold = vi
        for vj in v[:i + 1]:
            if vj < viold:
                vi -= 1
        n *= (-1)**vi
    return n


def matrixP(P):
    # horrible list comprehension that turns [1, 0] into [[0, 1], [1, 0]]
    return np.matrix([[(0, 1)[i == P[n]] for i in range(len(P))]
                      for n in range(len(P))])
