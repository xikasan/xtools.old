# coding: utf-8


def copy(sources, targets):
    return soft_update(sources, targets, 1.0)


def soft_update(sources, targets, tau):
    for source, target in zip(sources, targets):
        target.assign(tau * source + (1. - tau) * target)
