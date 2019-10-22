# coding: utf-8

import os
import datetime


# path correction
def go_to_root(root_name=None, verbose=False):
    cwd = os.getcwd()
    if root_name is not None:
        pos_root = cwd.find(root_name) + len(root_name)
    else:
        pos_workspace = cwd.find("workspace/") + len("workspace/")
        sub_cwd = cwd[pos_workspace:]
        pos_root = pos_workspace + sub_cwd.find("/")

    root_path = cwd[0:pos_root]
    os.chdir(root_path)

    if verbose:
        print("[info] cwd:", os.getcwd())


def join(*args):
    return os.path.join(*args)

def mkdirs(path, exist_ok=False):
    if not exist_ok and not os.path.exists(path):
        raise FileExistsError
    os.mkdirs(path, exist_ok)

def generate_time_dir_path(path=None, format="%Y.%m.%d.%H%M%S"):
    if path is None:
        path = os.getcwd()
    now = datetime.datetime.now().strftime(format)
    return join(path, now)

def make_dirs_current_time(path=None, format="%Y.%m.%d.%H%M%S", exist_ok=True):
    path = generate_time_dir_path(path)
    mkdirs(path, exist_ok)
    return path
