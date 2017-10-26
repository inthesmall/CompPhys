import numpy as np


def pivot(M, v):
    used = []
    outM = []
    outV = []
    for col in range(len(M)):
        entries = []
        for row in range(len(M)):
            if M[row, col] != 0:
                entries.append((M[row, col], row))

        if entries == []:
            raise ValueError("Col of 0's!")
        entries.sort(reverse=True)
        for entry in entries:
            if entry[1] not in used:
                used.append(entry[1])
                outM.append(M.A[entry[1]])
                outV.append(v[entry[1]])

                break
        else:
            unused = [i for i in range(len(M)) if i not in used]
            new = unused[0]
            old = entries[0][1]
            final = M.A[new] + M.A[old]
            used.append(new)
            outM[old] = final
            outV[old] = v[new] + v[old]
            outM.append(M.A[old])
            outV.append(v[old])

    return np.matrix(outM), np.array(outV)



