import cubic
import linear
import numpy as np
import matplotlib.pyplot as plt
import sys


def generate():
    xs = [-2.1, -1.45, -1.3, -0.2, 0.1, 0.15, 0.8, 1.1, 1.5, 2.8, 3.8]

    ys = [0.012155, 0.122151, 0.184520, 0.960789, 0.990050, 0.977751,
          0.527292, 0.298197, 0.105399, 3.936690e-4, 5.355348e-7]
    n_xs = np.linspace(xs[0], xs[-1], 200)
    c_ys = [cubic.spline_inter(x, xs, ys) for x in n_xs]

    l_ys = [linear.linint(x, xs, ys) for x in n_xs]

    plt.plot(xs, ys, 'bo', label="data")
    plt.plot(n_xs, l_ys, 'g-', label="Linear interpolation")
    plt.plot(n_xs, c_ys, 'r-', label="Cubic interpolation")
    plt.legend()
    plt.show()


def test1():
    xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.plot(xs, ys, 'ro')
    inter_xs = np.arange(0, 9, 0.1)
    inter_ys = [linear.linint(x, xs, ys) for x in inter_xs]
    plt.plot(inter_xs, inter_ys, 'b')
    plt.show()


def test2():
    xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ys = [0, 1, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5]
    plt.plot(xs, ys, 'ro')
    inter_xs = np.arange(0, 9, 0.1)
    inter_ys = [linear.linint(x, xs, ys) for x in inter_xs]
    plt.plot(inter_xs, inter_ys, 'b')
    plt.show()


if __name__ == '__main__':
    try:
        if sys.argv[1] == "1":
            test1()
        elif sys.argv[1] == "2":
            test2()
        else:
            generate()
    except IndexError:
        generate()
