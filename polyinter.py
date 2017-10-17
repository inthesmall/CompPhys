import math
import matplotlib.pyplot as plt


def gen(n):
    return [math.sin(i / 10) for i in range(n)]


def inter(fi, x):
    total = 0
    for i in range(len(fi)):
        prod = 1
        for j in range(len(fi)):
            if i == j:
                continue
            prod *= (x - j / 10) / (i / 10 - j / 10)
        total += prod * fi[i]
    return total


def test(n):
    l = len(n)
    interd = [inter(n, i / 100) for i in range(l * 10)]
    plt.plot([i / 100 for i in range(l * 10)], interd,
             'ro', [i / 10 for i in range(l)], n, 'bo')
    plt.show()
