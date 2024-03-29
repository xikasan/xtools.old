# coding: utf-8

import gym
import numpy as np
from cached_property import cached_property


class BaseModel:

    def __init__(self, dt, name="BaseModel", dtype=np.float32):
        self.dt = dt
        self.name = name
        self.dtype = dtype

    def __call__(self, action):
        raise NotImplementedError()

    @cached_property
    def action_size(self):
        return self.action_space.high.size

    @cached_property
    def observation_size(self):
        return self.observation_space.high.size

    @staticmethod
    def generate_space(low, high, dtype=np.float32):
        high = np.array(high) if type(high) is list else high
        low  = np.array(low)  if type(low)  is list else low
        high = high.astype(dtype)
        low  = low.astype(dtype)
        return gym.spaces.Box(high=high, low=low)

    @staticmethod
    def generate_inf_space(size, dtype=np.float32):
        high = np.array([ np.inf] * size).astype(dtype)
        low  = np.array([-np.inf] * size).astype(dtype)
        space = gym.spaces.Box(high, low)
        return high, low, space
