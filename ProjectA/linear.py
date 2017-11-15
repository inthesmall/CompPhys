def linint(x, xs, ys):
    """Linear interpolator

    Interpolates data given by *xs* and *ys* at the point *x*

    Params:
        x: float
            Point at which to interpolate function
        xs: list of floats
            x values of data
        ys: list of floats
            y values of data

    Returns:
        y: float
            Function interpolated at *x*
    """
    # Trivial case, just return the matching value
    if x in xs:
        return ys[xs.index(x)]
    # find i:
    try:
        i = [i >= x for i in xs].index(True) - 1
    except ValueError as err:
        # Raise an error if the value cannot be interpolated
        if err.args[0] == "True is not in list":
            raise ValueError("x is not in range of data") from err
        else:
            raise err
    if i == -1:
        raise ValueError("x is not in range of data")
    # Do the interpolation and return
    return (((xs[i + 1] - x) * ys[i] + (x - xs[i]) * ys[i + 1]) /
            (xs[i + 1] - xs[i]))
