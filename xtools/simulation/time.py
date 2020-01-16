# coding: utf-8

import numpy as np
import xtools as xt


def due_step_range(due, dt):
    return range(1, int(due / dt + 1))


def generate_step_time(due, dt, begin=0):
    return [xt.round(x, 2) for x in np.arange(begin+dt, due+dt, dt)]
