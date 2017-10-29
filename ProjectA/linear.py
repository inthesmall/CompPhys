def linint(xs, ys, x):
    # find i:
    try:
        i = [i > x for i in xs].index(True) - 1
    except ValueError as err:
        if err.args[0] == "True is not in list":
            raise ValueError("x is not in range of data") from err
        else:
            raise err
    if i == -1:
        raise ValueError("x is not in range of data")
    return (((xs[i + 1] - x) * ys[i] + (x - xs[i]) * ys[i + 1]) /
            (xs[i + 1] - xs[i]))
