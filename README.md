# ros_pybullet_drones

A ROS package containing a multi-drone gym environment `gym_bullet_drones` based on PyBullet

## Requirements and Installation

Our custom multi-drone env is built on `gym-pybullet-drones` module.

* For installing `gym-pybullet-drones` module, please follow https://github.com/utiasDSL/gym-pybullet-drones
* For installing our custom multi-drone gym environment
    ```
    cd ./gym_bullet_drones
    pip install -e .
    ```
* For installing 3D navigation goal tool plugin in rviz, please refer to https://github.com/BruceChanJianLe/rviz-3d-nav-goal-tool

## Usage

To run simple test of `gym_bullet_drones`

```
roslaunch ros_pybullet_drones run.launch
```

To run `gym_bullet_drones` with a simple flocking controller based on Boids rules

```
roslaunch ros_pybullet_drones run.launch use_flocking_controller:=true
```

* additional arguments
  * set `use_bullet_gui=true` to use bullet gui
  * set `num_drones` to specify the number of drones, default number is 10
  