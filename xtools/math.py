# coding: utf-8

import numpy as np


def d2r(d):
    if hasattr(d, "__len__"):
        return [d2r(de) for de in d]
    return d * np.pi / 180

def r2d(r):
    if hasattr(r, "__len__"):
        return [r2d(re) for re in r]
    return r * 180 / np.pi

def round(x, d=0):
    p = 10 ** d
    return float(np.floor((x * p) + np.copysign(0.5, x))) / p


# numpy
def as_ndarray(target):
    # print(target, type(target))
    if isinstance(target, np.ndarray):
        return target

    if isinstance(target, list) or isinstance(target, tuple):
        return np.array(target)

    if isinstance(target, dict):
        return np.array(list(target.values))

    if not hasattr(target, "__len__"):
        return np.array([target])

    raise ValueError("Not supported type")
