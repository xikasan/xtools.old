# coding: utf-8


def print_msg(kind, content, val=None):
    print("[{}]".format(kind), content, end="")
    if val is not None:
        print(":", val, end="")
    print("")


def info(content, val=None):
    print_msg("info", content, val=val)


def debug(content, val=None):
    print_msg("debug", content, val=val)

