import numpy as np
import yaml


class FlockingController(object):
    def __init__(self, config, env):
        with open(config) as f:
            self.params = yaml.load(f, Loader=yaml.FullLoader)
            self.env = env

    def plan(self, obs):
        state = np.array([uav_obs["state"] for uav_obs in obs])
        goal = np.array([uav_obs["goal"] for uav_obs in obs])
        pos = state[:, :3]
        vel = state[:, 7:10]
        adj_m, dist, sorted_inds = self._getAdjacencyMatrix(pos)
        vel_cmd = np.zeros((self.env.NUM_DRONES, 3))
        for k in range(self.env.NUM_DRONES):
            v_i = np.zeros(3) # interaction velocity
            v_a = np.zeros(3) # alignment velocity
            v_n = np.zeros(3) # navigation velocity
            n_nbrs = np.sum(adj_m[k, :])
            k_nbrs = self.params["k_nbrs"] if n_nbrs > self.params["k_nbrs"] else n_nbrs
            if k_nbrs > 0:
                k_inds = sorted_inds[k, :k_nbrs]
                v_i = 2 * (np.expand_dims(self.params["rs"]**2/dist[k, k_inds]**3 - 1/dist[k, k_inds], axis=1)) * (pos[k, :] - pos[k_inds, :])/dist[k, k_inds, np.newaxis] 
                v_i = np.mean(v_i, axis=0)
                vel_n = np.linalg.norm(vel, axis=1)
                non_zero_inds, = np.where(vel_n != 0)
                if len(non_zero_inds) > 0:
                    v_a = vel[non_zero_inds, :]/vel_n[non_zero_inds, np.newaxis]
                    v_a = np.mean(v_a, axis=0)
            v_i = self.params["ki"] * v_i
            v_a = self.params["ka"] * v_a
            if not np.any(np.isnan(goal[k, :])):
                v_n = self.params["k_nav"] * (goal[k, :] - pos[k, :])
            v = v_i + v_a + v_n
            if np.linalg.norm(v) > self.params["v_n"]:
                v = self.params["v_n"] * v/np.linalg.norm(v)
            vel_cmd[k, :] = v
        return vel_cmd

    def _getAdjacencyMatrix(self, pos):
        diff = np.reshape(pos, (self.env.NUM_DRONES, 1, 3)) - np.reshape(pos, (1, self.env.NUM_DRONES, 3))
        dist2 = np.multiply(diff[:, :, 0], diff[:, :, 0]) + np.multiply(diff[:, :, 1], diff[:, :, 1]) + np.multiply(diff[:, :, 2], diff[:, :, 2])
        np.fill_diagonal(dist2, np.inf) # ignore agent itself
        adjacency_mat = (dist2 < self.env.NEIGHBORHOOD_RADIUS * self.env.NEIGHBORHOOD_RADIUS).astype(np.float32).astype(int)
        sorted_inds = np.argsort(dist2)
        dist = np.sqrt(dist2)
        return adjacency_mat, dist, sorted_inds

        
if __name__ == "__main__":
    import gym
    import gym_bullet_drones
    num_drones = 5
    env = gym.make("bullet-drones-v0", num_drones=num_drones, use_gui=False, boundary_size=(40, 40, 6), init_formation="circle")
    flocking_controller = FlockingController("/home/jiawei/Projects/ROS_PROJECTS/catkin_ws_bullet/src/ros_pybullet_drones/config/flocking.yaml", env)
    env.setPlanner(flocking_controller)
    obs = env.reset()
    print(flocking_controller.params)
    vel_cmd = flocking_controller.plan(obs)
    print(vel_cmd)
