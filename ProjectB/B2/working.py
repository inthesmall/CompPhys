import numpy as np
import random


def _simpson(func, a, b, n):
    # This is actually 2h, but this way requires fewer operations
    h = (b - a) / n
    out = func(a)
    out += 2 * sum(func(np.arange(a + h, b, h)))
    out += 4 * sum(func(np.arange(a + h / 2, b + h / 2, h)))
    out *= h / 6
    return out


def simpson(func, a, b, eps):
    """
    Integrate *func* from *a* to *b* using simpsons rule to accuracy *eps*
    """
    I1 = _simpson(func, a, b, 1)
    I2 = _simpson(func, a, b, 2)
    n = 2
    while abs((I2 - I1) / I1) > eps:
        n *= 2
        I1 = I2
        I2 = _simpson(func, a, b, n)
    return I2


def trapeze(func, a, b, n):
    """
    Integrate *func* from *a* to *b* using the trapezium method with *n* slices
    """
    h = (b - a) / (n)
    i = np.linspace(a + h, b - h, n - 2)
    out = 0.5 * (func(a) + func(b))
    out += sum(func(i))
    out *= h
    return out


def monte(func, a, b, eps, dist=None):
    length = b - a
    if dist is None:
        def rand(n=1):
            return length * np.random.random(n) + a
    else:
        rand = generate_distribution(dist, a, b)
    r = rand()
    I1 = length * func(r) / dist(r)
    r = rand()
    I2 = (I1 + length * func(r)) / (2 * dist(r))
    n = 2
    while abs((I1 - I2) / I1) > eps:
        I1 = I2
        I2 = I1 * (n / (n + 1))
        n += 1
        r = rand()
        I2 += length * func(r) / (n * dist(r))
        print(I2)
    return I2


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


def monte(func, a, b, eps, dist=None):
    length = b - a
    if dist is None:
        def rand(n=1):
            return length * np.random.random(n) + a
    else:
        rand = generate_distribution(dist, a, b)
    r = rand()
    I1 = func(r) / dist(r)
    r = rand()
    I2 = (I1 + func(r)) / (2 * dist(r))
    n = 2
    while abs((I1 - I2) / I1) > eps:
        I1 = I2
        I2 = I1 * (n / (n + 1))
        n += 1
        r = rand()
        I2 += func(r) / (n * dist(r))
        print(I2)
    return I2


def a(n=1):
    return -(-0.537 + np.sqrt(0.537**2 - 2 * 0.139 *
                              np.random.random(n))) / 0.139


def b(x):
    return 0.537 - 0.139 * x


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
