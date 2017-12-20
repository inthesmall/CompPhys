import numpy as np
import random


def psi(z):
    # calculates |psi(z)|^2
    return (1 / np.sqrt(np.pi)) * np.exp(-(z**2))


def rand(n=1):
    return -(-0.98 + np.sqrt(0.98**2 - 2 * 0.48 *
                             np.random.random(n))) / 0.48


def dist(x):
    return 0.98 - 0.48 * x


def trapeze(func, a, b, eps):
    """
    Integrate *func* from *a* to *b* using trapezium rule to accuracy *eps*
    """
    h = b - a
    I1 = h * 0.5 * (func(a) + func(b))
    n = 0
    h /= 2
    I2 = I1 / 2 + h * func((a + b) / 2)
    while abs((I2 - I1) / I1) > eps:
        I1 = I2
        n += 1
        h /= 2
        I2 = I1 / 2 + h * sum(func(np.linspace(a + h, b - h, 2**n)))
    return I2


def simpson(func, a, b, eps):
    h = (b - a) / 2
    I1 = (func(a) + func(b) + 4 * func((a + b) / 2)) * (h / 3)
    I2 = I1 - (2 / 3) * h * func((a + b) / 2)
    I2 /= 2
    h /= 2
    I2 += (4 / 3) * h * sum((func(a + h), func(b - h)))
    n = 1
    while abs((I2 - I1) / I1) > eps:
        I1 = I2
        I2 = I1 - (2 / 3) * h * sum(func(np.linspace(a + h, b - h, 2**n)))
        I2 /= 2
        h /= 2
        n += 1
        I2 += (4 / 3) * h * sum(func(np.linspace(a + h, b - h, 2**n)))
    return I2


def monte(func, a, b, eps, rand=None, dist=None):
    length = b - a
    if dist is None:
        def rand(n=1):
            return length * np.random.random(n) + a

        I1 = length * func(rand())
        I2 = (I1 + length * sum(func(rand(9)))) / 10
        n = 10
        test = I1
        while abs((test - I2) / I2) > eps:
            # I1 = I2
            # I2 = I1 * (n / (n + 1))
            # n += 1
            # I2 += length * func(rand()) / n
            I1 = I2
            I2 = I1 / 10
            n *= 10
            I2 += length * sum(func(rand(int(0.9 * n)))) / n
            test = I2 * (n / (n + 1))
            test += length * func(rand()) / (n + 1)
        return I2[0]

    else:
        if rand is None:
            rand = generate_distribution(dist, a, b)
        r = rand()
        I1 = func(r) / (dist(r))
        r = rand(9)
        I2 = (I1 + sum(func(r) / dist(r)) / 10)
        n = 10
        test = I1
        while abs((test - I2) / I2) > eps:
            I1 = I2
            I2 = I1 / 10
            n *= 10
            r = rand(int(0.9 * n))
            I2 += sum(func(r) / dist(r)) / n
            test = I2 * (n / (n + 1))
            r = rand()
            test += func(r) / ((n + 1) * dist(r))
        return I2[0]


def generate_distribution(func, a, b):
    def rand(n=1):
        y = []
        i = 0
        while i < n:
            r = (random.random() * (b - a)) + a
            if random.random() <= func(r):
                y.append(r)
                i += 1
        return np.array(y)
    return rand


def adaptive(func, a, b, eps, bins=10):
    length = b - a
    h = length / bins
    weights = np.full(bins, 0.1)
    factors = np.zeros(bins)

    def make_rand(weights, h):
        cumul = np.cumsum(weights)

        def rand(n=1):
            xs = []
            for i in range(n):
                bin_ = np.where(cumul > np.random.random())[0][0]
                xs.append(h * bin_ + h * np.random.random())
            return np.array(xs)
        return rand

    def update_factors(r):
        bin_ = np.floor(r / h)
        factor = (func(r) * length) / (I2 * p(r)) - 1
        if isinstance(bin_, float):
            factors[int(bin_)] += factor
        else:
            bin_ = bin_.astype(int)
            for b, f in zip(bin_, factor):
                factors[b] += f

    def update_weights():
        nonlocal factors
        nonlocal weights
        nonlocal I2
        I2[0] *= sum(weights * h)
        factors /= sum(abs(factors))
        factors += 1
        for i in range(len(weights)):
            weights[i] *= factors[i]
        weights /= sum(weights)
        factors = np.zeros(bins)
        I2[0] /= sum(weights * h)

    def p(r):
        return np.array([weights[int(np.floor(ri / h))] for ri in r])

    rand = make_rand(weights, h)
    r = rand()
    I1 = length * func(r)
    I2 = I1
    update_factors(r)
    r = rand(9)
    I2 = (I1 + sum(func(r) / p(r)) / 10)
    update_factors(r)
    n = 10
    test = I1
    while abs((test - I2) / I2) > eps:
        update_weights()
        rand = make_rand(weights, h)
        I1 = I2
        I2 = I1 / 10
        n *= 10
        r = rand(int(0.9 * n))
        I2 += sum(func(r) / p(r)) / n
        update_factors(r)
        test = I2 * (n / (n + 1))
        r = rand()
        test += func(r) / ((n + 1) * p(r))
        update_factors(r)
    return I2[0] * sum(weights * h)
