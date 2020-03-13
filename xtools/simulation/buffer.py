# coding: utf-8

import numpy as np
import tensorflow as tf
import xtools as xt


def RLDataDict(model):
    return {
        "state": model.obs_size,
        "action": model.act_size,
        "next_state": model.obs_size,
        "reward": 1,
        "done": 1
    }


class Batch:

    def __init__(self, size):
        self.size = size


class ReplayBuffer:

    def __init__(self, keys_dims, capacity=36001, dtype=np.float32):
        self._buf = {key: np.empty(shape=(capacity, dim), dtype=dtype) for key, dim in keys_dims.items()}
        self.index = 0
        self.capacity = capacity
        self._is_full = False

    def __len__(self):
        return self.capacity if self._is_full else self.index+1

    def add(self, **kwargs):
        for key, val in kwargs.items():
            self._buf[key][self.index, :] = val

        if (self.index+1) == self.capacity:
            self._is_full = True
            self.index = 0
            return

        self.index += 1

    def sample(self, size):
        high = self.capacity if self._is_full else self.index
        idx = np.random.randint(0, high, size)
        return {
            key: vals[idx, :] for key, vals in self._buf.items()
        }

    def batch(self, size):
        batch_data = self.sample(size)
        batch = Batch(size)
        for key, val in batch_data.items():
            batch.__setattr__(key, val)
        return batch

    def _buffer(self):
        if self._is_full:
            return self._buf
        return {
            key: val[0:self.index] for key, val in self._buf.items()
        }

    def buffer(self):
        if not self._is_full:
            return {key: val[:self.index] for key, val in self._buf.items()}
        if self.index == 0:
            return self._buf
        return {
            key: np.concatenate(
                [val[self.index:], val[:self.index]], axis=0
            ) for key, val in self._buf.items()
        }


class TFReplayBuffer:

    def __init__(self, keys_dims, capacity=6001, dtype=tf.float32):
        self.dtyoe = dtype
        self._buf = {key: tf.Variable(np.zeros((capacity, dim)), dtype=dtype) for key, dim in keys_dims.items()}
        self.index = 0
        self.capacity = capacity
        self._is_full = False

    def __len__(self):
        return self.capacity if self._is_full else self.index

    def add(self, **keyvals):
        for key, val in keyvals.items():
            # format
            if not isinstance(val, tf.Tensor):
                val = xt.as_ndarray(val)
            if isinstance(val, tf.Tensor) and len(val.shape) > 1:
                val = tf.squeeze(val, axis=1)

            self._buf[key][self.index].assign(val)
        if (self.index+1) == self.capacity:
            self._is_full = True
            self.index = 0
            return

        self.index += 1

    def sample(self, size):
        high = self.capacity if self._is_full else self.index
        idx = tf.random.uniform([size], minval=0, maxval=high, dtype=tf.int32)
        return {key: tf.gather(val, idx, axis=0) for key, val in self._buf.items()}

    def batch(self, size):
        batch_data = self.sample(size)
        batch = Batch(size)
        for key, val in batch_data.items():
            batch.__setattr__(key, val)
        return batch

    def _buffer(self):
        if self._is_full:
            return self._buf
        return {
            key: val[0:self.index] for key, val in self._buf.items()
        }

    def buffer(self):
        if not self._is_full:
            {key: val[:]}


class ListReplayBuffer:

    def __init__(self, keys, capacity=36001, dtype=np.float32):
        self._buf = {key: [None]*capacity for key in keys}
        self.index = 0
        self.capacity = capacity
        self._is_full = False

    def __len__(self):
        return self.capacity if self._is_full else self.index+1

    def add(self, **kwargs):
        for key, val in kwargs.items():
            self._buf[key][self.index] = val

        if (self.index+1) == self.capacity:
            self._is_full = True
            self.index = 0
            return

        self.index += 1

    def sample(self, size):
        high = self.capacity if self._is_full else self.index
        idx = np.random.randint(0, high, size)
        return {
            key: vals[idx, :] for key, vals in self._buf.items()
        }

    def batch(self, size):
        batch_data = self.sample(size)
        batch = Batch(size)
        for key, val in batch_data.items():
            batch.__setattr__(key, val)
        return batch

    def buffer(self):
        if self._is_full:
            return self._buf
        return {
            key: vals[0:self.index] for key, vals in self._buf.items()
        }


class Retriever:

    def __init__(self, source):
        self._source = source

    def __call__(self, name, idx=None):
        temp = self._source[name]
        if idx is None:
            return np.squeeze(temp)
        return np.squeeze(temp[:, idx])


class SimpleReplayBuffer:

    def __init__(self):
        self.storage = []

    def add(self, data):
        self.storage.append(data)

    def sample(self, batch_size=100):
        idxs = np.random.randint(0, len(self.storage), batch_size)
        x, u, y, r, d = [], [], [], [], []

        for i in idxs:
            X, U, Y, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            u.append(np.array(U, copy=False))
            y.append(np.array(Y, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(X), np.array(U), np.array(Y), np.array(R).reshape(-1, 1), np.array(D).reshape(-1, 1),
