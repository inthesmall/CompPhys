import linear
import matplotlib.pyplot as plt
import numpy as np
import sys


def test1():
    xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.plot(xs, ys, 'ro')
    inter_xs = np.arange(0, 9, 0.1)
    inter_ys = [linear.linint(xs, ys, x) for x in inter_xs]
    plt.plot(inter_xs, inter_ys, 'b')
    plt.show()


def test2():
    xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ys = [0, 1, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5]
    plt.plot(xs, ys, 'ro')
    inter_xs = np.arange(0, 9, 0.1)
    inter_ys = [linear.linint(xs, ys, x) for x in inter_xs]
    plt.plot(inter_xs, inter_ys, 'b')
    plt.show()


if sys.argv[1] == "1":
    test1()
elif sys.argv[1] == "2":
    test2()