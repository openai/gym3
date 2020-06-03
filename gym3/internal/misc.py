def intprod(xs):
    """
    Product of a sequence of integers
    """
    out = 1
    for x in xs:
        out *= x
    return out


def allsame(xs):
    """
    Returns whether all elements of sequence are the same
    """
    assert len(xs) > 0
    return all(x == xs[0] for x in xs[1:])
