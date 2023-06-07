#!/usr/bin/env python

import gym
import time
import numpy as np
import rospy
import gym_bullet_drones
from flocking_controller import FlockingController

from signal import signal, SIGINT

np.set_printoptions(precision=4, suppress=True)


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

    num_drones = rospy.get_param("/bullet_drones_node/num_drones", 10)
    use_bullet_gui = rospy.get_param("/bullet_drones_node/use_bullet_gui", False)
    planner_type = rospy.get_param("/bullet_drones_node/planner_type", "flocking_controller")

    if planner_type == "simple_planner":
        env = gym.make("bullet-drones-v0", num_drones=num_drones, use_gui=use_bullet_gui, boundary_size=(40, 40, 6), init_formation="circle")
        planner = Planner(num_drones=num_drones)
        env.setPlanner(planner)

        obs = env.reset()

        t0 = time.time()
        for i in range(100):
            a = env.plan(obs)
            next_obs, reward, done, info = env.step(a)
            obs = next_obs
            # print("[TIME] {} s".format(time.time()-t0))

        obs = env.reset()
        for i in range(100):
            a = env.plan(obs)
            next_obs, reward, done, info = env.step(a)
            obs = next_obs
            # print("[TIME] {} s".format(time.time()-t0))
        t1 = time.time()

        print("[SIMULATION TIME] {} s".format(t1 - t0))

        env.close()
    elif planner_type == "flocking_controller":
        env = gym.make("bullet-drones-v0", num_drones=num_drones, use_gui=use_bullet_gui, boundary_size=(40, 40, 6), init_formation="circle", normalize_obs=False,
                       takeoff_height=3.0)
        flocking_controller = FlockingController("/home/jiawei/Projects/ROS_PROJECTS/catkin_ws_bullet/src/ros_pybullet_drones/config/flocking.yaml", env)
        env.setPlanner(flocking_controller)

        obs = env.reset()

        t0 = time.time()
        for i in range(1000):
            a = env.plan(obs)
            next_obs, reward, done, info = env.step(a)
            obs = next_obs

        t1 = time.time()

        print("[SIMULATION TIME] {} s".format(t1 - t0))


    
