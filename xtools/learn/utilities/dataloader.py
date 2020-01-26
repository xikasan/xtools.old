# coding: utf-8

import numpy as np
import pandas as pd
import xtools as xt
from pathlib import Path


class Batch:

    def __init__(self, size):
        self.size = size
        self.keys = []

    def set(self, key, val):
        val = xt.as_ndarray(val)
        if hasattr(self, key):
            stored = self.__getattribute__(key)
            stored = np.concatenate([stored, val], axis=0)
            self.__setattr__(key, stored)
            return
        # new keyword
        self.keys.append(key)
        self.__setattr__(key, val)


class DataLoader:

    def __init__(self,
                 batch_size,
                 at_random=True,
                 file_name=None,
                 directory_name=None,
                 list_name=None,
                 exist_header=True,
                 dtype=np.float32):
        self.dtype = dtype
        self._batch_size = batch_size
        self._at_random = at_random
        self._header = "infer" if exist_header else None
        self._data = {}
        if file_name is not None:
            self.load_from_file(file_name)
        elif directory_name is not None:
            self.load_from_directory(directory_name)
        elif list_name is not None:
            self.load_from_list(list_name)
        else:
            raise ValueError("file_name or directory_name or list_name must be given")
        self.size = len(self._data[list(self._data.keys())[0]])
        self._indices = list(range(self.size))
        self._current = 0
        self._num_batch = int(np.ceil(self.size / self._batch_size))

    def __iter__(self):
        self._current = 0
        if self._at_random:
            np.random.shuffle(self._indices)
        return self

    def __next__(self):
        if self._current >= self.size:
            raise StopIteration
        idx = self._indices[self._current:self._current+self._batch_size]
        self._current += self._batch_size
        batch = Batch(self._batch_size)
        for key, vals in self._data.items():
            batch.set(key, vals[idx])
        return batch

    def __len__(self):
        return self._num_batch

    def load_from_file(self, file_name):
        data = pd.read_csv(file_name, header=self._header)
        self._data = {key: val.values for key, val in data.iteritems()}

    def load_from_directory(self, directory_name):
        data = Path(directory_name).glob("*.csv")
        data = [str(d) for d in data]
        data = [pd.read_csv(d, header=self._header) for d in data]
        data = pd.concat(data, axis=0, ignore_index=True)
        self._data = {key: val.values for key, val in data.iteritems()}

    def load_from_list(self, data_list):
        with open(data_list, "r") as fp:
            data = fp.readlines()
        data = [d.strip() for d in data]
        data = list(filter(lambda x: not x == "", data))
        data = [pd.read_csv(d, header=self._header) for d in data]
        data = pd.concat(data, axis=0, ignore_index=True)
        self._data = {key: val.values for key, val in data.iteritems()}
