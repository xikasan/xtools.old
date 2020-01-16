# coding: utf-8

import numpy as np
import xtools as xt
import xtools.simulation as xs
# from xaircraft.models.base import BaseModel
from xtools.simulation.models.model import BaseModel


class Filter1st(BaseModel):

    def __init__(self, dt, tau, init_value=0.0, name="Filter1st"):
        super().__init__(dt, name=name)
        self.tau = tau
        self._x = init_value
        self._dx = 0.0
        self._init_val = init_value

    def __call__(self, source):
        fn = lambda x: (source - self._x) / self.tau
        self._dx = xs.no_time_rungekutta(fn, self.dt, self._x)
        self._x += self._dx * self.dt
        return self.get_state()

    def reset(self):
        self._x = self._init_val
        self._dx = 0
        return self.get_state()

    def get_state(self):
        return self._x

    def get_full_state(self):
        return np.array([self._x, self._dx])


class Filter2nd(BaseModel):

    def __init__(self, dt, tau, init_value=0.0, name="Filter2nd", **kwargs):
        super().__init__(dt, name=name, **kwargs)
        self.tau = tau
        self._A, self._B = self._construct_matrix(tau)
        self._x = np.array([0, init_value], dtype=self.dtype)
        self._dx = np.array([0, 0], dtype=self.dtype)
        self._init_val = init_value

    def __call__(self, source):
        fn = lambda x: x.dot(self._A) + source * self._B
        self._dx = xs.no_time_rungekutta(fn, self.dt, self._x)
        self._x += self.dt * self._dx
        return self.get_state()

    def reset(self):
        self._x = np.array([0, self._init_val], dtype=self.dtype)
        self._dx = np.array([0, 0], dtype=self.dtype)
        return self.get_state()

    def get_state(self):
        return self._x[0]

    def get_full_state(self):
        return np.concatenate([self._x, self._dx[1:]])

    def _construct_matrix(self, tau):
        it = 1 / tau
        it2 = it * it
        A = np.array([
            [0, 1],
            [-it2, -2 * it]
        ], dtype=self.dtype).T
        B = np.array([0, it2], dtype=self.dtype)
        return A, B
