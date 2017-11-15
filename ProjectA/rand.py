import math
import random
import time


def create_a():
    # Ensure consistent results by always using the same seed
    random.seed(a=1)
    # Create 1e5 numbers
    r = [random.random() for x in range(100000)]
    return r


def create_b():
    # Start with uniformly distributed numbers
    r = create_a()
    # Shift them using the inverse of the CDF
    y = [math.acos(-2 * (x - 0.5)) for x in r]
    return y


def create_c():

    random.seed(a=1)

    # initialize the output list
    y1 = []

    # Using this instead of comparing with len(y1) results in a 10% speedup
    i = 0
    # Keep going until we have enough numbers
    while i < 100000:
        # Domain is 0 to pi
        r = random.random() * math.pi
        # This is singificantly faster than
        # r = random.uniform(0, math.pi)
        # and is good enough for what we need

        # Generate a new random number and compare to the probability of
        # getting r
        # This is random < P(r), where P is the desired PDF normalized to have
        # a max value of one.
        if random.random() < math.sin(r)**2:
            # If this comparison number is smaller than P(r) then we accept
            y1.append(r)
            i += 1

    return y1


def time_ratio():
    """Time the functions create_b() and create_c() and return their ratio

    Returns how many times longer create_c() takes to run than create_b()
    """
    t1 = time.process_time()
    create_b()
    t2 = time.process_time()
    b = t2 - t1

    t1 = time.process_time()
    create_c()
    t2 = time.process_time()
    c = t2 - t1

    # Would expect something along the lines of 2 times as long.
    # Only accept 50% of generated values -> double
    # It is likely to be higher than this, since we are generating two
    # randoms for each run, and also perfoming more operations such as
    # comparisons.
    #
    # Line-by-line runtime analysis shows that 200067 are generated, which
    # is almost exactly twice 1e5. This function returns 3.7±0.3 on my
    # machine.
    return c / b
