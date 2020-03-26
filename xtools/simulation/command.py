# coding: utf-8

import numpy as np
import xtools as xt


class BaseCommand:

    def __init__(self, amplitude=1.0, bias=None, name="Command"):
        self._amplitude = amplitude
        self._bias = bias
        self.name = name
        self._zero = np.zeros_like(amplitude)

    def __call__(self, time):
        raise NotImplementedError()


class StepCommand(BaseCommand):

    def __init__(self, step_time, name="StepCommand", **kwargs):
        super().__init__(name=name, **kwargs)
        self._step_time = step_time

    def __call__(self, time):
        return self._amplitude if time >= self._step_time else 0.0


class RectangularCommand(BaseCommand):

    def __init__(self, period, name="RectangularCommand", **kwargs):
        super().__init__(name=name, **kwargs)
        self._period = float(period)

    def __call__(self, time):
        return xt.pulse(time, self._period, self._amplitude, bias=self._bias)


class RandomRectangularCommand(BaseCommand):

    def __init__(self, max_amplitude, interval, name="RandomRectangularCommand", **kwargs):
        super().__init__(name=name, **kwargs)

        assert isinstance(interval, (int, float)) and interval > 0, "interval must be a positive number"
        self._max_amplitude = max_amplitude
        self._rate = 1.0 / interval
        self._next_time = 0.
        self._generate_next_time()

    def __call__(self, time):
        if time >= self._next_time:
            self._generate_next_target()
            self._generate_next_time()
        return self._amplitude

    def _generate_next_time(self):
        self._next_time += -np.log(1 - np.random.rand()) / self._rate

    def _generate_next_target(self):
        self._amplitude = (np.random.rand() * 2 - 1) * self._max_amplitude

    def reset(self):
        self._next_time = 0.0
        self._generate_next_time()
        self._generate_next_target()


if __name__ == '__main__':
    dt = 0.1
    due = 10
    amp = 1
    period = 2
    command = RandomRectangularCommand(1, 1)

    import xtools.simulation as xs
    log = xs.ReplayBuffer({"time": 1, "cmd": 1}, capacity=int(due/dt+1))
    log.add(time=0.0, cmd=0.0)
    for time in xs.generate_step_time(due, dt):
        cmd = command(time)
        log.add(time=time, cmd=cmd)
        xt.print_msg(
            "{:5.2f}".format(time),
            "cmd: {:3.1f}".format(cmd)
        )

    import pandas as pd
    import matplotlib.pyplot as plt
    result = log.buffer()
    result = xs.Retriever(result)
    result = pd.DataFrame({
        "time": result("time"),
        "cmd": result("cmd")
    })
    result.plot(x="time", y="cmd")
    plt.show()
