# A Multi-Drone Gym Environment `gym_bullet_drones`

## Documentation

The environment uses Bullet physics engines to simulate flight dynamics of multiple quadrotors. It also includes several optional custom disturbance forces, such as air drags, downwash effects and ground effects. The quadrotor parameters are defined in an `.urdf` file. The default quadrotor model is Bitcraze Craziflie 2.0 in the X configuration. Custom quadrotor models are possible by user-defined `.urdf` files.

When the gym envrionment is created, a daemon thread will be spawned to call the backend Bullet physics engine to simulate the quadrotor dynamics at 240 Hz. The flight controller is customizable to take different reference commands and output desired rpms. The default flight controller is a PID controller that takes velocity commands and it runs at 48 Hz. 

The user can use the env method `setPlanner` to set custom planner, e.g., conventional flocking controller or neural network policy. The custom planner should implement a public method `plan` which generates an action based on the current observation. The planner runs at 10 Hz by default.

* `BulletDronesEnv`: the base class of the multi-drone gym environment based on PyBullet and ROS
  * **Observation space**: A `Dict` which contains `state` and `goal`
  * **Action space**: A 3D velocity command
  * Private methods
    * `_computeObs(self)`: computes the observations of all agents, given as a list of individual `Dict` observations; to be implemented by user based on RL problem formulation
    * `_computeReward(self)`: to be implemented by user based on RL problem formulation
    * `_computeDone(self)`: to be implemented by user based on RL problem formulation
    * `_computeInfo(self)`: to be implemented by user if necessary
  * Public methods
    * `reset(self)`: reset the environment and return the list of individual observations
    * `step(self, action)`: action is defined as a numpy array of shape `(NUM_DRONES, 3)`, and returns a tuple `(obs, reward, done, info)`
    * `close(self)`: exit the simulation daemon thread and disconnect the Bullet engine
* `DepthBulletDronesEnv`: the derived class of `BulletDronesEnv` which supports onboard depth perception
  * To be implemented and integrated with Unity through [ROS TCP endpoint](https://github.com/Unity-Technologies/ROS-TCP-Endpoint)