# coding: utf-8

import yaml


# config
def load_config(file):
    with open(file) as fp:
        conf = yaml.safe_load(fp)
    return conf
