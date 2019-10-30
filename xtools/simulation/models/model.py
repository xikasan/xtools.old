# coding: utf-8

import numpy as np
import tensorflow as tf

tk = tf.keras


class DefaultModel:

    def __init__(self, dt, name="DefaultModel", dtype=tf.float32):
        self.dt = dt
        self.name = name
        self.dtype = dtype

        self.__variable_counter = 0

    def __call__(self, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def new_variable(self, value, name=None, dtype=None):
        dtype = dtype if dtype is not None else self.dtype
        name = name if name is not None else self.name+"_var-{}".format(self.__variable_counter)
        self.__variable_counter += 1
        return tf.Variable(value, dtype=dtype, name=name) * 1
