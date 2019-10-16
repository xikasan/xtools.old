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
