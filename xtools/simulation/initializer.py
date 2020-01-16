# coding: utf-8

import numpy as np


class DefaultInitializer:

    def __init__(self, name, size):
        self.name = name
        self.size = size

    def get(self):
        raise NotImplementedError()


class ZeroInitializer(DefaultInitializer):

    def __init__(self, size):
        super().__init__("ZeroInitializer", size)

    def get(self):
        return np.zeros(self.size)


class UniformInitializer(DefaultInitializer):

    def __init__(self, size, range=[0, 1]):
        super().__init__("RandomUniformInitializer", size)
        self._range = range

    def get(self):
        return np.random.uniform(*self._range, self.size)


class NormalInitializer(DefaultInitializer):

    def __init__(self, size, mean=0, var=1):
        super().__init__("UniformInitializer", size)
        self._mean = mean
        self._var = var

    def get(self):
        return np.random.normal(self._mean, self._var, self.size)


if __name__ == '__main__':
    initializer = NormalInitializer(3, mean=10, var=0.01)
    print(initializer.get())
