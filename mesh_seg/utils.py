from itertools import tee


def pairwise(iterable):
    """Iterate over all pairs of consecutive items in an iterable.

    Notes
    -----
        [s0, s1, s2, s3, ...] -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def groupwise(iterable, group_size=2):
    """Iterate over all groups of consecutive items in an iterable.

    Notes
    -----
        groupwise([s0, s1, s2, s3, ...], group_size=2) -> (s0, s1), (s2, s3), ...
        groupwise([s0, s1, s2, s3, ...], group_size=3) -> (s0, s1, s2) (s3, s4, s5), ...
    """
    iterable = iter(iterable)
    while True:
        try:
            yield tuple(next(iterable) for _ in range(group_size))
        except RuntimeError:
            return
