# Python only implements doubles. Therefore use numpy, as it has more types
import numpy as np
# Calculate epsilon for 16, 32, 64 and 128 bit


def half_precision():
    # set up variables to use. Using np.float# to specify precision
    float_one = np.float16(1.0)
    float_delta = np.float16(1.0)
    # Halve epsilon until 1+epsilon is no longer distinguishable from 1'
    while float_one + float_delta != float_one:
        float_delta /= np.float16(2)
    # Double it again to get the smallest distinguishable from one
    float_delta *= np.float16(2)
    print("Half precision epsilon", float_delta)


def single_precision():
    float_one = np.float32(1.0)
    float_delta = np.float32(1.0)
    while float_one + float_delta != float_one:
        float_delta /= np.float32(2)
    float_delta *= np.float32(2)
    print("Single precision epsilon: ", float_delta)


def double_precision():
    float_one = np.float64(1.0)
    float_delta = np.float64(1.0)
    while float_one + float_delta != float_one:
        float_delta /= np.float64(2)
    float_delta *= np.float64(2)
    print("Double precision epsilon: ", float_delta)


def extended_precision():
    # this is implemented on my laptop but not on the uni PCs.
    # if it is not implemented then it will throw an error
    float_one = np.float128(1.0)
    float_delta = np.float128(1.0)
    while float_one + float_delta != float_one:
        float_delta /= np.float128(2)
    float_delta *= np.float128(2)
    print("Extended precision epsilon: ", float_delta)


def validate_method():
    # Ensure that a smaller suitable value cannot be found
    # Use half precision floats
    float_one = np.float16(1.0)
    float_delta = np.float16(1.0)
    while float_one + float_delta != float_one:
        float_delta /= np.float16(2)
    float_delta *= np.float16(2)
    # Boolean value. Is there rounding? Is 1+epsilon actually equal to the
    # value of one plus epsilon
    valid = (np.float128((float_delta) + float_one)
             == (np.float128(float_delta)) + 1.)
    # make epsilon slightly smaller
    float_delta *= np.float16(0.999)
    # Boolean. Does our slightly smaller epsilon work?
    invalid = (np.float128((float_delta) + float_one)
               == (np.float128(float_delta)) + 1.)
    if valid and not invalid:
        # Let's not be too confident...
        print("The method has not been shown to be invalid")
    else:
        print("Method invalid")


if __name__ == '__main__':
    validate_method()
    half_precision()
    single_precision()
    double_precision()
    try:
        extended_precision()
    except Exception:
        print("Float 128 not implemented. You are probably on Windows")
