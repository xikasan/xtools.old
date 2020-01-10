# coding: utf-8


import numpy as np


def pulse(time, period, amplitude, bias=None):
    ret = amplitude if (time % period) < (period / 2) else -amplitude
    if bias is not None:
        ret += bias
    return ret
