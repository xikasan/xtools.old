# coding: utf-8

import xtools as xt
from xtools.learn.utilities.dataloader import DataLoader

xt.go_to_root()

list_name = "tests/dummy.txt"

loader = DataLoader(5, list_name=list_name)
print(type(loader))
print(len(loader))

exit()

for batch in loader:
    print(batch.time, batch.w1)
    exit()
