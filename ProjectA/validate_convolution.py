import convolution
import matplotlib.pyplot as plt
import numpy as np


def h(t):
    if t >= 2 and t <= 4:
        return 5
    else:
        return 0


def g(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)


hs = [h(i) for i in np.linspace(-10, 10, 2**16)]
gs = [g(i) for i in np.linspace(-10, 10, 2**16)]
xs = [i for i in np.linspace(-10, 10, 2**16)]
h = hs.copy()
g = gs.copy()
x = xs.copy()

xc, c = convolution.convolve_list(xs, gs, xs, hs)

assert hs == h
assert gs == g
assert xs == x

plt.plot(xs, gs, 'r-', label="y=g(x)")
plt.plot(xs, hs, 'g-', label="y=h(x)")
plt.plot(xc, c, 'b-', label="Convolution y=g(x)*h(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
