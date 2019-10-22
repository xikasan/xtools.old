# coding: utf-8

import os


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
