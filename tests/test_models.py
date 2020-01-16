# coding: utf-8

import unittest
import tensorflow as tf
from xtools.simulation.models.model import DefaultModel


class TestDefaultModel(unittest.TestCase):

    def test_new_variable(self):
        model_name = "test"
        var_name = "var"

        model = DefaultModel(1/50, name=model_name)
        var_1 = model._new_variable(1)
        self.assertEqual(var_1.dtype, tf.float32)
        self.assertEqual(var_1.numpy(), 1.)


if __name__ == '__main__':
    unittest.main()
