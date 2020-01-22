# coding: utf-8

import numpy as np
import xtools as xt
from xtools import simulation as xsim
from xtools.simulation.models.model import *
tk = tf.keras


class Vehicle(BaseModel):
    """
    Implementation of a linear model of a ground vehicle in x-y world.
    Motion is represented with the forward velocity v(t), and heading angle theta(t), and the position (x, y).
    Velocity range is between 0 and v_max( = 5 in default), and heading is -180 ~ 180 deg.
    Theta = 0 means that vehicle is heading to x-axis, and counter-clockwise is positive.
    dv/dt = (v_cmd - v) / tau_v
    dT/dt = (T_cmd - T) / tau_T
    dx/dt = v cos(T)
    dy/dt = v sin(T)
    """


    IX_X = 0
    IX_Y = 1
    IX_V = 2
    IX_T = 3

    def __init__(
            self,
            *args,
            max_velocity=5,
            init_position=xsim.ZeroInitializer(2),
            range_position=[[-np.inf, np.inf], [-np.inf, np.inf]],
            tau_v=0.25,
            tau_T=0.50,
            **kwargs
    ):
        if "name" not in kwargs.keys():
            kwargs["name"] = "Vehicle"
        super().__init__(*args, **kwargs)

        self.init_position = init_position

        self._x = self.__init_state()
        self._tau_v = tau_v
        self._tau_T = tau_T
        self._A, self._B = self.__construct_params()

        # action space
        self.act_low  = np.array([0, -np.pi])
        self.act_high = np.array([max_velocity, np.pi])
        self.action_space = self.generate_space(self.act_low, self.act_high)

        # observation space
        self.obs_low  = np.array([np.min(range_position[0]), np.min(range_position[1]), 0, -np.pi])
        self.obs_high = np.array([np.max(range_position[0]), np.max(range_position[1]), max_velocity, np.pi])
        self.observation_space = self.generate_space(self.obs_low, self.obs_high)

    def __call__(self, action):
        pass

    def reset(self):
        self._x = self.__init_state()
        return self.get_obs()

    def __init_state(self):
        position = self.init_position.get()
        vT = np.zeros(2)
        return np.concatenate([position, vT], axis=0)

    def __construct_params(self):
        av = 1 / self._tau_v if not self._tau_v < 1e-5 else 1e-5
        aT = 1 / self._tau_T if not self._tau_T < 1e-5 else 1e-5

        def A(T):
            cos = np.cos(T)
            sin = np.sin(T)
            return np.array([
                [0, 0, cos, 0],
                [0, 0, sin, 0],
                [0, 0, -av, 0],
                [0, 0, 0, -aT]
            ]).T
        B = np.array([
            [ 0,  0],
            [ 0,  0],
            [av,  0],
            [ 0, aT],
        ]).T
        return A, B

    def step(self, action):
        def f(x):
            return x.dot(self._A(x[self.IX_T])) + action.dot(self._B)
        dx = xsim.no_time_rungekutta(f, self.dt, self._x) * self.dt
        next_x = self._x + dx
        next_x[self.IX_T] = self.__round_theta(next_x[self.IX_T])
        self._x = np.clip(next_x, self.obs_low, self.obs_high)
        return self.get_obs()

    def get_state(self):
        return self._x

    def get_obs(self):
        return self.get_state()

    @staticmethod
    def __round_theta(theta):
        if theta > np.pi:
            return Vehicle.__round_theta(theta - (np.pi * 2))
        if theta < -np.pi:
            return Vehicle.__round_theta(theta + (np.pi * 2))
        return theta


if __name__ == '__main__':
    model = Vehicle(1/50, name="Vehicle", range_position=[[-10, 10], [-10, 10]])
    daction = np.array([10, 0])
    print("="*60)
    for t in range(50 * 10):
        print("step: {:04}".format(t), "state:", model.step(daction))

    # dummy_act = model.new_variable([1, 2])
    # for _ in range(10):
    #     model.step(dummy_act)
