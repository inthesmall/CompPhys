import numpy as np
import random


def psi(z):
    # calculates |psi(z)|^2
    return (1 / np.sqrt(np.pi)) * np.exp(-(z**2))


def rand(n=1):
    # Helper function which generates the distribution needed for importance
    # sampling.
    return -(-0.98 + np.sqrt(0.98**2 - 2 * 0.48 *
                             np.random.random(n))) / 0.48


def dist(x):
    # Distribution for importance sampling
    return 0.98 - 0.48 * x


def trapeze(func, a, b, eps):
    """
    Integrate *func* from *a* to *b* using trapezium rule to accuracy *eps*
    """
    # Initial estimate
    h = b - a
    I1 = h * 0.5 * (func(a) + func(b))
    n = 0
    h /= 2
    # Update estimate with a third point
    I2 = I1 / 2 + h * func((a + b) / 2)
    # Keep halving the step-size until precision is reached
    while abs((I2 - I1) / I1) > eps:
        I1 = I2
        n += 1
        h /= 2
        # Update the estimate. The linspace is the best way I could find to
        # generate the points to sample. Gives 1/4,3/4; 1/8,3/8,5/8,7/8;...
        I2 = I1 / 2 + h * sum(func(np.linspace(a + h, b - h, 2**n)))
    return I2


def simpson(func, a, b, eps):
    """
    Integrate *func* from *a* to *b* using simpson's rule to accuracy *eps*
    """
    # The project sheet suggested using 4/3 - 1/3 trapezium rule trick, but I
    # found that this evaluated faster
    h = (b - a) / 2
    # Inital estimate from three points
    I1 = (func(a) + func(b) + 4 * func((a + b) / 2)) * (h / 3)
    # Change the 4/3 factor to a 2/3 factor ready for the next step
    I2 = I1 - (2 / 3) * h * func((a + b) / 2)
    I2 /= 2
    h /= 2
    # Add the points in between and update the estimate
    I2 += (4 / 3) * h * sum((func(a + h), func(b - h)))
    n = 1
    # Keep going until we have the precision we need
    while abs((I2 - I1) / I1) > eps:
        I1 = I2
        # Again, turn the midpoints of quadratics into endpoints by turning
        # 4/3 into 2/3
        I2 = I1 - (2 / 3) * h * sum(func(np.linspace(a + h, b - h, 2**n)))
        I2 /= 2
        h /= 2
        n += 1
        # Add the 4/3 midpoints
        I2 += (4 / 3) * h * sum(func(np.linspace(a + h, b - h, 2**n)))
    return I2


def monte(func, a, b, eps, rand=None, dist=None):
    """
    Integrate *func* from *a* to *b* using Monte Carlo to accuracy *eps*

    Can specify *rand* and *dist* to implement importance sampling.
    Both must be vectorizable (able to operate on an np.array of data)

    Integral of *dist* from *a* to *b* should be 1. *rand* should generate
    values distributed according to *dist*

    It is possible to specify only *dist*, in which case a very slow rejection
    method will be used to create *rand*. This should be avoided if possible.
    """
    length = b - a
    if dist is None:
        # Uniform deviate
        def rand(n=1):
            return length * np.random.random(n) + a

        # First estimate from a single sample
        I1 = length * func(rand())
        # Second estimate from 10 samples
        I2 = (I1 + length * sum(func(rand(9)))) / 10
        n = 10
        test = I1
        while abs((test - I2) / I2) > eps:
            I1 = I2
            I2 = I1 / 10
            # If there are 10 times the number of samples at each iteration,
            # it is much less likely that the function converges as a fluke.
            # It is easier to be cautious than to keep track of a rolling
            # average epsilon. This gives far better results than doubling at
            # each loop
            n *= 10
            I2 += length * sum(func(rand(int(0.9 * n)))) / n
            # We need a single step to calculate epsilon
            test = I2 * (n / (n + 1))
            test += length * func(rand()) / (n + 1)
        return I2[0]

    else:
        # Importance Sampling
        if rand is None:
            # Make a rand function if we just have dist
            rand = generate_distribution(dist, a, b)
        r = rand()
        # We divide by dist(r) to normalise our results against the weighting
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
    # This is an adapted version of my rejection method from Project A
    # We can use it to generate variables according to a general distribution
    # given by func
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
    """
    Integrate *func* from *a* to *b* using adaptive MC method to accuracy *eps*

    *bins* gives the number of pieces to split the function into in order to
    create the importance distribution. It will be a piecewise flat function in
    *bins* sections.
    """
    length = b - a
    h = length / bins
    # Initialize distribution as uniform
    weights = np.full(bins, 1 / bins)
    factors = np.zeros(bins)

    def make_rand(weights, h):
        # Find the CDF
        cumul = np.cumsum(weights)

        def rand(n=1):
            xs = []
            for i in range(n):
                bin_ = np.where(cumul > np.random.random())[0][0]
                xs.append(h * bin_ + h * np.random.random())
            return np.array(xs)
        return rand

    def update_factors(r):
        # Record the proportion of func(r) to the mean and put it in the
        # appropriate bin
        bin_ = np.floor(r / h)
        factor = (func(r) * length) / (I2 * p(r)) - 1
        if isinstance(bin_, float):
            factors[int(bin_)] += factor
        else:
            bin_ = bin_.astype(int)
            for b, f in zip(bin_, factor):
                factors[b] += f

    def update_weights():
        # Take the factors we recorded and update the PDF with them
        nonlocal factors
        nonlocal weights
        nonlocal I2
        factors /= sum(abs(factors))
        factors += 1
        for i in range(len(weights)):
            weights[i] *= factors[i]
        # Renormalise
        weights /= sum(weights)
        # Reset factors to zero ready for the next time
        factors = np.zeros(bins)

    def p(r):
        # convenience function which returns value of PDF at r
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
    # Same as in the other Monte Carlo, except here we update factors every
    # time we generate random numbers, and update the PDF and rand function
    # once at the start of every loop
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
