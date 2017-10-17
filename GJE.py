class Matrix():
    def __init__(self, data):
        self.rows = []
        self.dim = len(data)
        for row in data:
            if len(row) != self.dim:
                raise ValueError("Must be a square matrix")
            self.rows.append(row)

    def __repr__(self):
        return str(self.rows)

    def get_rows(self):
        return self.rows

    def __mul__(self, other):
        if isinstance(other, Vector):
            return NotImplemented


class Vector():
    def __init__(self, data):
        self.data = data
        self.dim = len(data)

    def __repr__(self):
        return str(self.data)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Matrix):
            if self.dim != other.dim:
                raise ValueError("Incompatible dimensions")
            else:
                ret = Vector([0] * self.dim)
                for i in range(self.dim):
                    for j in range(self.dim):
                        ret.data[i] += other.get_rows()[i][j] * self.data[j]
                return ret
        else:
            return NotImplemented


def GJE(M, v):
    # Do pivoting
    A = Matrix(M.get_rows())
    b = Vector(v.data)
    for i in range(A.dim):
        b.data[i] /= A.get_rows()[i][i]
        A.get_rows()[i] = [el / A.get_rows()[i][i] for el in A.get_rows()[i]]
        for j in range(A.dim):
            if i == j:
                pass
            else:
                factor = A.get_rows()[j][i]
                b.data[j] -= factor * b.data[i]
                for k in range(A.dim):
                    A.get_rows()[j][k] -= factor * A.get_rows()[i][k]
    return A, b


def pivot(M):
    A = Matrix(M.get_rows())
    for j in range(A.dim):
        for i in range(A.dim):
            if A.get_rows()[i][j] != 0:
                break
        else:
            raise Exception("Col of 0's!")