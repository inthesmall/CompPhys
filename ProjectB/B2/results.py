from main import *
import numpy as np
import timeit
import sys


I = 0.497661132509476367081034628184


def fast():
    print("Trapezium Method estimate")
    print(trapeze(psi, 0, 2, 1e-6))
    print("It took")
    print("{:3.0f}µs".format(timeit.timeit(
        "trapeze(psi, 0, 2, 1e-6)", number=100, globals=globals()) * 1e4))
    print("Simpson's Rule estimate")
    print(simpson(psi, 0, 2, 1e-6))
    print("It took")
    print("{:3.0f}µs".format(timeit.timeit(
        "simpson(psi, 0, 2, 1e-6)", number=100, globals=globals()) * 1e4))
    print("Flat Monte Carlo estimate")
    print(monte(psi, 0, 2, 1e-6))
    print("It took")
    print("{:.0f}ms".format(timeit.timeit(
        "monte(psi, 0, 2, 1e-6)", number=10, globals=globals()) * 100))
    print("Monte Carlo estimate with linear importance sampling")
    print(monte(psi, 0, 2, 1e-6, rand, dist))
    print("It took")
    print("{:.0f}ms".format(timeit.timeit(
        "monte(psi, 0, 2, 1e-6)", number=10, globals=globals()) * 100))
    print("Monte Carlo estimate with adaptive importance sampling")
    print(adaptive(psi, 0, 2, 1e-6))
    print("It took")
    print("{:.2f} seconds".format(timeit.timeit(
        "adaptive(psi, 0, 2, 1e-6)", number=3, globals=globals()) / 3))


def slow():
    print("This may take a few minutes")
    tra = timeit.repeat("trapeze(psi, 0, 2, 1e-6)",
                        number=100, globals=globals(), repeat=3)
    sim = timeit.repeat("simpson(psi, 0, 2, 1e-6)",
                        number=100, globals=globals(), repeat=3)
    flat = timeit.repeat("monte(psi, 0, 2, 1e-6)",
                         number=10, globals=globals(), repeat=3)
    lin = timeit.repeat("monte(psi, 0, 2, 1e-6, rand, dist)",
                        number=10, globals=globals(), repeat=3)
    ada = timeit.repeat("adaptive(psi, 0, 2, 1e-6)",
                        number=3, globals=globals(), repeat=3)
    tram = np.mean(tra) * 1e4
    tras = np.std(tra) * 1e4
    trae = trapeze(psi, 0, 2, 1e-6)
    simm = np.mean(sim) * 1e4
    sims = np.std(sim) * 1e4
    sime = simpson(psi, 0, 2, 1e-6)
    flatm = np.mean(flat) * 100
    flats = np.std(flat) * 100
    flate = monte(psi, 0, 2, 1e-6)
    linm = np.mean(lin) * 100
    lins = np.std(lin) * 100
    line = monte(psi, 0, 2, 1e-6, rand, dist)
    adam = np.mean(ada) / 3
    adas = np.std(ada)
    adae = adaptive(psi, 0, 2, 1e-6)

    print("Trapezium Method estimate")
    print("{} ({:+0.2g}%)".format(trae, ((trae - I) / I) * 100))
    print("It took {:3.0f}±{:2.0f}µs".format(tram, tras))
    print("Simpson's Rule estimate")
    print("{} ({:+0.2g}%)".format(sime, ((sime - I) / I) * 100))
    print("It took {:3.0f}±{:2.0f}µs".format(simm, sims))
    print("Flat Monte Carlo estimate")
    print("{} ({:+0.3f}%)".format(flate, ((flate - I) / I) * 100))
    print("It took {:3.0f}±{:2.0f}ms".format(flatm, flats))
    print("Monte Carlo estimate with linear importance sampling")
    print("{} ({:+0.3f}%)".format(line, ((line - I) / I) * 100))
    print("It took {:3.0f}±{:2.0f}ms".format(linm, lins))
    print("Monte Carlo estimate with adaptive importance sampling")
    print("{} ({:+0.3f}%)".format(adae, ((adae - I) / I) * 100))
    print("It took {:3.1f}±{:2.1f} seconds".format(adam, adas))


a = 0


def count_(eps, psi):
    global a
    a = 0
    trapeze(psi, 0, 2, eps)
    tra = a
    a = 0
    simpson(psi, 0, 2, eps)
    sim = a
    a = 0
    monte(psi, 0, 2, eps)
    flat = a
    a = 0
    monte(psi, 0, 2, eps, rand, dist)
    lin = a
    a = 0
    adaptive(psi, 0, 2, eps)
    ada = a
    return tra, sim, flat, lin, ada


def count():
    global a
    print("counting...")

    def psi(z):
        # I understand this is frowned upon, but it is the easiest way to count
        # function evaluations without modifying the algorithms, and I am less
        # likely to miss a call this way. On balance I feel it is justfied.
        global a
        if isinstance(z, float) or isinstance(z, int):
            a += 1
        else:
            a += len(z)
        return (1 / np.sqrt(np.pi)) * np.exp(-(z**2))
    for eps in [1e-3, 1e-4, 1e-5, 1e-6]:
        tra, sim, flat, lin, ada = count_(eps, psi)
        print("Epsilon =", eps)
        print("Trapezium Method estimate evaluated the function\
 {} times".format(tra))
        print("Simpson's Rule estimate evaluated the function\
 {} times".format(sim))
        print("Flat Monte Carlo estimate evaluated the function\
 {} times".format(flat))
        print("Monte Carlo estimate with linear importance sampling evaluated\
 the function {} times".format(lin))
        print("Monte Carlo estimate with adaptive importance sampling evaluated\
 the function {} times".format(ada))


def help():
    print("Generate results for use in project B2")
    print("Options:")
    print("    -f: fast. Evalute the function, time it once")
    print("    -s: slow. Evalute the function, time it three times, do stats")
    print("        This takes around 3 minutes")
    print("    -c: count. Count the number of integrand function calls.")


try:
    if sys.argv[1] == "-s":
        slow()
    elif sys.argv[1] == "-c":
        count()
    elif sys.argv[1] == "-h" or sys.argv[1] == "--help" or sys.argv[1] == "help":
        help()
    else:
        help()
except IndexError:
    help()
