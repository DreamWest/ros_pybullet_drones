#!/usr/bin/env python

import gym
import time
import numpy as np
import gym_bullet_drones

from signal import signal, SIGINT


def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    exit(0)


class Planner(object):
    def __init__(self, nn_policy=None, num_drones=1):
        self.nn_policy = nn_policy
        self.num_drones = num_drones

    def plan(self, obs=None):
        if self.nn_policy is None:
            return np.array([[0.0, 1.0, 0.0] for _ in range(self.num_drones)])


if __name__ == "__main__":
    signal(SIGINT, handler)

    num_drones = 20
    env = gym.make("bullet-drones-v0", num_drones=num_drones, use_gui=False, boundary_size=(40, 40, 6), init_formation="circle")
    planner = Planner(num_drones=num_drones)
    env.setPlanner(planner)

    obs = env.reset()

    t0 = time.time()
    for i in range(50):
        a = env.plan(obs)
        next_obs, reward, done, info = env.step(a)
        obs = next_obs
        # print("[TIME] {} s".format(time.time()-t0))

    obs = env.reset()
    for i in range(50):
        a = env.plan(obs)
        next_obs, reward, done, info = env.step(a)
        obs = next_obs
        # print("[TIME] {} s".format(time.time()-t0))
    t1 = time.time()

    print("[SIMULATION TIME] {} s".format(t1 - t0))

    env.close()

    
