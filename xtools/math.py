# coding: utf-8

import numpy as np


def d2r(d):
    if hasattr(d, "__len__"):
        if isinstance(d, list):
            return [d2r(de) for de in d]
        if isinstance(d, tuple):
            return (d2r(de) for de in d)
    return d * np.pi / 180


def r2d(r):
    if hasattr(r, "__len__"):
        if isinstance(r, list):
            return [r2d(re) for re in r]
        if isinstance(r, tuple):
            return (r2d(re) for re in r)
    return r * 180 / np.pi


def __round(x, d=0):
    p = 10 ** d
    return float(np.floor((x * p) + np.copysign(0.5, x))) / p


def round(x, d=0):
    if hasattr(x, "__len__"):
        if isinstance(x, list):
            return [
                round(xi, d) for xi in x
            ]
        if isinstance(x, tuple):
            return [
                round(xi, d) for xi in x
            ]
        if isinstance(x, np.ndarray):
            return np.array([
                round(xi, d) for xi in x
            ])
        raise ValueError("Not supported type: {} is given.".format(type(x)) )
    return __round(x, d)


# numpy
def as_ndarray(target):
    if isinstance(target, np.ndarray):
        return target

    if isinstance(target, list) or isinstance(target, tuple):
        return np.array(target)

    if isinstance(target, dict):
        return np.array(list(target.values))

    if not hasattr(target, "__len__"):
        return np.array([target])

    raise ValueError("Not supported type")
