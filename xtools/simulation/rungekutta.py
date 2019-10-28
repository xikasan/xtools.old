# coding: utf-8

def rungekutta(f, dt, t0, x0):
    dt2 = dt / 2
    k1 = f(t0, x0)
    k2 = f(t0 + dt2, x0 + dt2 * k1)
    k3 = f(t0 + dt2, x0 + dt2 * k2)
    k4 = f(t0 + dt , x0 + dt  * k3)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6

def no_time_rungekutta(fn, dt, x0):
    dt2 = dt / 2
    k1 = fn(x0)
    k2 = fn(x0 + dt2 * k1)
    k3 = fn(x0 + dt2 * k2)
    k4 = fn(x0 + dt  * k3)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6
