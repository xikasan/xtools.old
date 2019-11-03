# coding: utf-8

import gym
import numpy as np
from xtools import simulation as xsim
from xtools.simulation.models.vehicle import Vehicle


class VehicleTravel(gym.Env):

    def __init__(
            self,
            range_world_x=[-100, 100],
            range_world_y=[-100, 100],
            init_target=xsim.UniformInitializer
    ):
        super().__init__()

        self.viewer = None

        self._range_world_x = np.array(range_world_x)
        self._range_world_y = np.array(range_world_y)
        self._vehicle = Vehicle(
            1/10,
            max_velocity=3,
            range_position=[range_world_x, range_world_y],
            init_position=xsim.ZeroInitializer(2)
        )

        self._target_initializer = init_target(2, self._range_world_x/2)
        self._target_position = None

        self.action_space = self._vehicle.action_space
        self.observation_space = self._vehicle.observation_space

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        obs = self._vehicle.step(action)
        err = self._target_position - obs[:2]
        loss = np.linalg.norm(err)
        reward = loss * (-1)

        return self.get_obs(), reward, False, []

    def reset(self):
        self._target_position = self._target_initializer.get()
        return self.get_obs()

    def get_obs(self):
        return np.concatenate([self._vehicle.get_obs(), self._target_position], axis=0)

    def render(self, mode="human"):
        screen_width = 800
        screen_height = 800
        screen_size = np.array([screen_width, screen_height])

        world_width = self._range_world_x[1] - self._range_world_x[0]
        world_height = self._range_world_y[1] - self._range_world_y[0]
        world_size = np.array([world_width, world_height])
        scale_x = screen_width / world_width
        scale_y = screen_height / world_height
        scale = np.array([scale_y, scale_y])

        vehicle_width = 2.0 * scale_x
        vehicle_length = 4.0 * scale_x

        target_pos = self._target_position * scale + screen_size / 2

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = -vehicle_width/2, vehicle_width/2, -vehicle_length/2, vehicle_length/2
            vehicle = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            vehicle.set_color(.8, .3, .3)
            self.vehicle_trans = rendering.Transform()
            vehicle.add_attr(self.vehicle_trans)
            self.viewer.add_geom(vehicle)

            target = rendering.make_circle(scale_x)
            target.set_color(.3, .8, .3)
            target.add_attr(rendering.Transform(translation=target_pos))
            self.viewer.add_geom(target)

        x, y, v, T = self._vehicle.get_obs()
        x *= scale_x
        y *= scale_y
        self.vehicle_trans.set_rotation(T + np.pi/2)
        self.vehicle_trans.set_translation(x+screen_width/2, y+screen_height/2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    import time
    env = VehicleTravel()
    env.reset()

    daction = np.array([5, np.pi/2])

    for step in range(10*20):
        action = env.action_space.sample()

        env.step(action)
        env.render()
        # print(step)
        time.sleep(1/50)

    env.close()
