#! /usr/bin/env python

import time
from typing import Any
import rospy
import gym
import gym.spaces as spaces
import numpy as np
import threading
import xml.etree.ElementTree as etxml
import pkg_resources
import pybullet as p
import pybullet_data
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation as R


MAX_LIN_VEL_XY = 1 
MAX_LIN_VEL_Z = 0.5
BOUNDARY_OFFSET = 0.5


class BulletDronesEnv(gym.Env):
    """The class of multi-UAV env based on PyBullet and ROS"""
    def __init__(self,
                 drone_model=DroneModel.CF2X,
                 num_drones=1,
                 neighborhood_radius=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 takeoff_height=0.5,
                 physics=Physics.PYB,
                 sim_freq=240,
                 control_freq=48,
                 plan_freq=10,
                 use_gui=False,
                 boundary_size=(20, 20, 3),
                 init_formation="circle",
                 **kwargs
                 ):
        """
        Parameters
        ----------
        drone_model: DroneModel, optional
            The desired drone type
        TODO: define kwargs
        TODO: complete the docstring
        """
        rospy.init_node("bullet_drones_env_node")

        # constants
        self.G = 9.8
        self.RAD2DEG = 180/np.pi
        self.DEG2RAD = np.pi/180
        self.SIM_FREQ = sim_freq
        self.TIMESTEP = 1./sim_freq
        self.CONTROL_FREQ = control_freq
        self.CONTROL_TIMESTEP = 1./control_freq
        self.CONTROL_INTERVAL = int(self.SIM_FREQ/self.CONTROL_FREQ)
        self.PLAN_FREQ = plan_freq
        self.PLAN_TIMESTEP = 1./plan_freq
        # params
        self.NUM_DRONES = num_drones
        self.NEIGHBORHOOD_RADIUS = neighborhood_radius
        self.TAKEOFF_HEIGHT = takeoff_height
        self.planner = None
        self.curr_plan_time = 0.
        self.action = np.zeros(3) # vel_cmd
        self.curr_goal = np.zeros(3)
        self.rpms = np.zeros((self.NUM_DRONES, 4))
        self.last_rpms = np.zeros((self.NUM_DRONES, 4))
        XLIM, YLIM, ZLIM = boundary_size
        self.boundary = np.array([[-XLIM/2.+BOUNDARY_OFFSET, -YLIM/2.+BOUNDARY_OFFSET, BOUNDARY_OFFSET], [XLIM/2.-BOUNDARY_OFFSET, YLIM/2.-BOUNDARY_OFFSET, ZLIM-BOUNDARY_OFFSET]])
        # options
        self.DRONE_MODEL = drone_model
        self.USE_GUI = use_gui
        self.PHYSICS = physics
        self.URDF = self.DRONE_MODEL.value + ".urdf"
        # load the drone properties from the .urdf file
        self.M, \
        self.L, \
        self.THRUST2WEIGHT_RATIO, \
        self.J, \
        self.J_INV, \
        self.KF, \
        self.KM, \
        self.COLLISION_H,\
        self.COLLISION_R, \
        self.COLLISION_Z_OFFSET, \
        self.MAX_SPEED_KMH, \
        self.GND_EFF_COEFF, \
        self.PROP_RADIUS, \
        self.DRAG_COEFF, \
        self.DW_COEFF_1, \
        self.DW_COEFF_2, \
        self.DW_COEFF_3 = self._parseURDFParameters()
        rospy.loginfo("[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
            self.M, self.L, self.J[0,0], self.J[1,1], self.J[2,2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO, self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3))
        # additional constants
        self.GRAVITY = self.G*self.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4*self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
        self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        elif self.DRONE_MODEL in [DroneModel.CF2P, DroneModel.HB]:
            self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        # https://www.chegg.com/homework-help/questions-and-answers/mainly-two-kinds-quadrotor-uav-market-plus-type-x-type-different-local-coordinate-result-d-q104557496
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)        
        # bullet setup
        if self.USE_GUI:
            self.CLIENT = p.connect(p.GUI)
            # disable rgb, depth, segmentation view
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=8,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[0, 0, 0],
                                         physicsClientId=self.CLIENT
                                         )
        else:
            self.CLIENT = p.connect(p.DIRECT)
            # TODO: implement recording camera setup
        # set initial poses
        if initial_xyzs is None:
            if init_formation == "circle":
                r = self.NUM_DRONES * self.L * 4
                self.INIT_XYZS = np.vstack([np.array([r*np.cos(2*np.pi*x/self.NUM_DRONES) for x in range(self.NUM_DRONES)]), \
                                            np.array([r*np.sin(2*np.pi*y/self.NUM_DRONES) for y in range(self.NUM_DRONES)]), \
                                            np.ones(self.NUM_DRONES) * self.TAKEOFF_HEIGHT]).transpose().reshape(self.NUM_DRONES, 3)
            elif init_formation == "square": # TODO: to figure out the correct logic for square formation
                n = np.ceil(np.sqrt(self.NUM_DRONES))

                self.INIT_XYZS = np.vstack([np.array([(x%n-(n-1)/2)*4*self.L for x in range(self.NUM_DRONES)]), \
                                            np.array([(-y//n+(n-1)/2)*4*self.L for y in range(self.NUM_DRONES)]), \
                                            np.ones(self.NUM_DRONES) * self.TAKEOFF_HEIGHT]).transpose().reshape(self.NUM_DRONES, 3)
        elif initial_xyzs.shape == (self.NUM_DRONES, 3):
            self.INIT_XYZS = initial_xyzs
        else:
            raise Exception("[ERROR] initial_xyzs should be in shape of (NUM_DRONES, 3)")
        if initial_rpys is None:
            self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        elif initial_rpys.shape == (self.NUM_DRONES, 3):
            self.INIT_RPYS = initial_rpys
        else:
            raise Exception("[ERROR] initial_rpys should be in shape of (NUM_DRONES, 3)")
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        # observation space and action space
        self.observation_space = self._observation_space()
        self.action_space = self._action_space()
        self.true_action_space = self._true_action_space()
        # set controller
        self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
        # ros setup
        self.sim_sleeper = rospy.Rate(self.SIM_FREQ)
        self.pose_pubs = [rospy.Publisher(f"/drone{i}/global/nwu_pose", PoseStamped, queue_size=10) for i in range(1, self.NUM_DRONES+1)]
        self.pose_msgs = [PoseStamped() for _ in range(self.NUM_DRONES)]
        self.odom_pubs = [rospy.Publisher(f"/drone{i}/global/nwu_odom", Odometry, queue_size=10) for i in range(1, self.NUM_DRONES+1)]
        self.odom_msgs = [Odometry() for _ in range(self.NUM_DRONES)]
        self.pose_odom_pub_timer = rospy.Timer(rospy.Duration(1./30), self._pose_odom_pub_callback) #TODO: to check if multiple timers are required
        self.drone_mesh_pub = rospy.Publisher("/drone_visualization", MarkerArray, queue_size=10)
        self.mesh_msgs = [Marker() for _ in range(self.NUM_DRONES)] #TODO: check whether marker array can work
        self.mesh_array_msg = MarkerArray()
        self.drone_mesh_pub_timer = rospy.Timer(rospy.Duration(1./30), self._drone_mesh_pub_callback)

        for i in range(self.NUM_DRONES):
            self.mesh_msgs[i].scale.x = 0.4
            self.mesh_msgs[i].scale.y = 0.4
            self.mesh_msgs[i].scale.z = 0.4

            self.mesh_msgs[i].color.r = 0.0
            self.mesh_msgs[i].color.g = 0.0
            self.mesh_msgs[i].color.b = 1.0
            self.mesh_msgs[i].color.a = 1.0

            self.mesh_msgs[i].mesh_resource = "package://ros_pybullet_drones/gym_bullet_drones/gym_bullet_drones/meshes/hummingbird.mesh"

        # background bullet sim thread
        self.sim_step = 0
        self.bullet_sim_mutex = threading.Lock()
        self.bullet_sim_condition = threading.Condition()
        self.is_resetting = True
        self.bullet_sim_thread = threading.Thread(target=self._run_sim, daemon=True)
        self.bullet_sim_thread.start()
        self.running = True
        self.stop_flag = False

    def _run_sim(self):
        """Thread function of running bullet simulation in the background"""
        rospy.logwarn("START BULLET SIMULATION THREAD!")
        while not rospy.is_shutdown():
            with self.bullet_sim_condition:
                if self.stop_flag:
                    self.running = False
                    self.bullet_sim_condition.notify_all()
                    break

                while self.is_resetting: # wait until resetting is finished
                    self.bullet_sim_condition.wait()

            with self.bullet_sim_mutex:
                if self.sim_step == self.CONTROL_INTERVAL:
                    self.rpms = self._apply_control(self.action)
                    self.sim_step = 0
                for i in range(self.NUM_DRONES):
                    self._physics(self.rpms[i, :], i)
                    # add external disturbances
                    if self.PHYSICS == Physics.PYB_GND:
                        self._groundEffect(self.rpms[i, :], i)
                    elif self.PHYSICS == Physics.PYB_DRAG:
                        self._drag(self.last_rpms[i, :], i)
                    elif self.PHYSICS == Physics.PYB_DW:
                        self._downwash(i)
                    elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                        self._groundEffect(self.rpms[i, :], i)
                        self._drag(self.last_rpms[i, :], i)
                        self._downwash(i)
                self.last_rpms = self.rpms
                p.stepSimulation(physicsClientId=self.CLIENT)
                self.sim_step += 1
                self._updateStateInformation()

            self.sim_sleeper.sleep()


    def _pose_odom_pub_callback(self, event):
        with self.bullet_sim_mutex:
            timestamp = rospy.Time.now()
            for i in range(self.NUM_DRONES):
                # pose
                self.pose_msgs[i].header.stamp = timestamp
                self.pose_msgs[i].header.frame_id = "map" #TODO: to change to world add static tf
                self.pose_msgs[i].pose.position.x = self.pos[i, 0] #TODO: to replace this with helper functions
                self.pose_msgs[i].pose.position.y = self.pos[i, 1]
                self.pose_msgs[i].pose.position.z = self.pos[i, 2]
                self.pose_msgs[i].pose.orientation.x = self.quat[i, 0]
                self.pose_msgs[i].pose.orientation.y = self.quat[i, 1]
                self.pose_msgs[i].pose.orientation.z = self.quat[i, 2]
                self.pose_msgs[i].pose.orientation.w = self.quat[i, 3]
                self.pose_pubs[i].publish(self.pose_msgs[i])
                # odom
                self.odom_msgs[i].header.stamp = timestamp
                self.odom_msgs[i].header.frame_id = "map"
                self.odom_msgs[i].pose.pose.position.x = self.pos[i, 0]
                self.odom_msgs[i].pose.pose.position.y = self.pos[i, 1]
                self.odom_msgs[i].pose.pose.position.z = self.pos[i, 2]
                self.odom_msgs[i].pose.pose.orientation.x = self.quat[i, 0]
                self.odom_msgs[i].pose.pose.orientation.y = self.quat[i, 1]
                self.odom_msgs[i].pose.pose.orientation.z = self.quat[i, 2]
                self.odom_msgs[i].pose.pose.orientation.w = self.quat[i, 3]
                self.odom_msgs[i].twist.twist.linear.x = self.vel[i, 0]
                self.odom_msgs[i].twist.twist.linear.y = self.vel[i, 1]
                self.odom_msgs[i].twist.twist.linear.z = self.vel[i, 2]
                self.odom_msgs[i].twist.twist.angular.x = self.ang_v[i, 0]
                self.odom_msgs[i].twist.twist.angular.y = self.ang_v[i, 1]
                self.odom_msgs[i].twist.twist.angular.z = self.ang_v[i, 2]
                self.odom_pubs[i].publish(self.odom_msgs[i])

    def _drone_mesh_pub_callback(self, event):
        with self.bullet_sim_mutex:
            timestamp = rospy.Time.now()
            for i in range(self.NUM_DRONES):
                self.mesh_msgs[i].header.stamp = timestamp
                self.mesh_msgs[i].header.frame_id = "map"

                self.mesh_msgs[i].id = i + 1
                self.mesh_msgs[i].type = Marker.MESH_RESOURCE
                self.mesh_msgs[i].action = Marker.ADD

                self.mesh_msgs[i].pose.position.x = self.pos[i, 0]
                self.mesh_msgs[i].pose.position.y = self.pos[i, 1]
                self.mesh_msgs[i].pose.position.z = self.pos[i, 2]

                if self.DRONE_MODEL == DroneModel.CF2X:
                    rpy = np.copy(self.rpy[i, :])
                    rpy[2] += np.pi/4
                    r = R.from_euler('xyz', rpy)
                    q = r.as_quat()

                    self.mesh_msgs[i].pose.orientation.x = q[0]
                    self.mesh_msgs[i].pose.orientation.y = q[1]
                    self.mesh_msgs[i].pose.orientation.z = q[2]
                    self.mesh_msgs[i].pose.orientation.w = q[3]
                elif self.DRONE_MODEL == DroneModel.CF2P:
                    self.mesh_msgs[i].pose.orientation.x = self.quat[i, 0]
                    self.mesh_msgs[i].pose.orientation.y = self.quat[i, 1]
                    self.mesh_msgs[i].pose.orientation.z = self.quat[i, 2]
                    self.mesh_msgs[i].pose.orientation.w = self.quat[i, 3]

            self.mesh_array_msg.markers = self.mesh_msgs
            self.drone_mesh_pub.publish(self.mesh_array_msg)

    def _observation_space(self):
        """Define observation space of a single drone

        current observation: {
            "state": [pos, quat, vel],
            "goal": goal_pos
        }

        Another choice is to include normalized angular velocity in the observation (to be done)
        current observation: {
            "state": [pos, quat, vel, ang_vel],
            "goal": goal_pos
        }

        # TODO: 1. to add different choices of representations
                2. to add other alternative representations like HER
        """
        return spaces.Dict(
            {
                "state": spaces.Box(
                    low=np.array([-1., -1., 0., -1., -1., -1., -1., -1., -1., -1.]),
                    high=np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]),
                    dtype=np.float32
                ),
                "goal": spaces.Box(
                    low=np.array([-1., -1., 0.]),
                    high=np.array([1., 1., 1.]),
                    dtype=np.float32
                )
            }
        )

    def _action_space(self):
        """Define action space of a single drone

        action shape: (3,)
        """
        return spaces.Box(
            low=np.array([-1., -1., -1.]),
            high=np.array([1., 1., 1.]),
            dtype=np.float32
        )
    
    def _true_action_space(self):
        """Define true action space of a single drone

        action shape: (3,)
        """
        return spaces.Box(
            low=np.array([-MAX_LIN_VEL_XY, -MAX_LIN_VEL_XY, -MAX_LIN_VEL_Z]),
            high=np.array([MAX_LIN_VEL_XY, MAX_LIN_VEL_XY, MAX_LIN_VEL_Z]),
            dtype=np.float32
        ) 

    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.
        """
        self.RESET_TIME = time.time()
        self.X_AX = -1*np.ones(self.NUM_DRONES)
        self.Y_AX = -1*np.ones(self.NUM_DRONES)
        self.Z_AX = -1*np.ones(self.NUM_DRONES)
        # initialize the drones kinemaatic information
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        # Set PyBullet's parameters
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        # Load ground plane, drone and obstacles models
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)
        self.DRONE_IDS = np.array([p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF),
                                              self.INIT_XYZS[i,:],
                                              p.getQuaternionFromEuler(self.INIT_RPYS[i,:]),
                                              flags = p.URDF_USE_INERTIA_FROM_FILE,
                                              physicsClientId=self.CLIENT
                                              ) for i in range(self.NUM_DRONES)])
        # TODO: add obstacles

    def _physics(self,
                 rpm,
                 nth_drone
                 ):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        """
        forces = np.array(rpm**2)*self.KF
        torques = np.array(rpm**2)*self.KM
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.CLIENT
                                 )
        p.applyExternalTorque(self.DRONE_IDS[nth_drone],
                              4,
                              torqueObj=[0, 0, z_torque],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.CLIENT
                              )
        
    def _groundEffect(self,
                      rpm,
                      nth_drone
                      ):
        """PyBullet implementation of a ground effect model.

        Inspired by the analytical model used for comparison in (Shi et al., 2019).

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        # Kin. info of all links (propellers and center of mass)
        link_states = np.array(p.getLinkStates(self.DRONE_IDS[nth_drone],
                                               linkIndices=[0, 1, 2, 3, 4],
                                               computeLinkVelocity=1,
                                               computeForwardKinematics=1,
                                               physicsClientId=self.CLIENT
                                               ))
        # Simple, per-propeller ground effects
        prop_heights = np.array([link_states[0, 0][2], link_states[1, 0][2], link_states[2, 0][2], link_states[3, 0][2]])
        prop_heights = np.clip(prop_heights, self.GND_EFF_H_CLIP, np.inf)
        gnd_effects = np.array(rpm**2) * self.KF * self.GND_EFF_COEFF * (self.PROP_RADIUS/(4 * prop_heights))**2
        if np.abs(self.rpy[nth_drone,0]) < np.pi/2 and np.abs(self.rpy[nth_drone,1]) < np.pi/2:
            for i in range(4):
                p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     i,
                                     forceObj=[0, 0, gnd_effects[i]],
                                     posObj=[0, 0, 0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT
                                     )
                
    def _drag(self,
              rpm,
              nth_drone
              ):
        """PyBullet implementation of a drag model.

        Based on the the system identification in (Forster, 2015).

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        # Rotation matrix of the base
        base_rot = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        # Simple draft model applied to the base/center of mass
        drag_factors = -1 * self.DRAG_COEFF * np.sum(np.array(2*np.pi*rpm/60))
        drag = np.dot(base_rot, drag_factors*np.array(self.vel[nth_drone, :]))
        p.applyExternalForce(self.DRONE_IDS[nth_drone],
                             4,
                             forceObj=drag,
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME,
                             physicsClientId=self.CLIENT
                             )
        
    def _downwash(self,
                  nth_drone
                  ):
        """PyBullet implementation of a ground effect model.

        Based on experiments conducted at the Dynamic Systems Lab by SiQi Zhou.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        for i in range(self.NUM_DRONES):
            delta_z = self.pos[i, 2] - self.pos[nth_drone, 2]
            delta_xy = np.linalg.norm(np.array(self.pos[i, 0:2]) - np.array(self.pos[nth_drone, 0:2]))
            if delta_z > 0 and delta_xy < 10: # Ignore drones more than 10 meters away
                alpha = self.DW_COEFF_1 * (self.PROP_RADIUS/(4*delta_z))**2
                beta = self.DW_COEFF_2 * delta_z + self.DW_COEFF_3
                downwash = [0, 0, -alpha * np.exp(-.5*(delta_xy/beta)**2)]
                p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     4,
                                     forceObj=downwash,
                                     posObj=[0, 0, 0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT
                                     )

    def _updateStateInformation(self):
        """Updates and stores the drones state information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).
        """
        for i in range (self.NUM_DRONES):
            self.pos[i], self.quat[i] = p.getBasePositionAndOrientation(self.DRONE_IDS[i], physicsClientId=self.CLIENT)
            self.rpy[i] = p.getEulerFromQuaternion(self.quat[i])
            self.vel[i], self.ang_v[i] = p.getBaseVelocity(self.DRONE_IDS[i], physicsClientId=self.CLIENT)

    def _parseURDFParameters(self):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.
        """
        URDF_TREE = etxml.parse(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF)).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
               GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3
    
    def _computeObs(self):
        """Returns the current observation of the environment.

        Normalize the observation w.r.t. the global frame
        """
        obs = []
        for i in range(self.NUM_DRONES):
            norm_pos = np.multiply(np.divide(self.pos[i, :]-self.boundary[0, :], self.boundary[1, :]-self.boundary[0, :]), 
                                   self.observation_space["state"].high[:3]-self.observation_space["state"].low[:3]) + self.observation_space["state"].low[:3]
            norm_goal = np.multiply(np.divide(self.curr_goal-self.boundary[0, :], self.boundary[1, :]-self.boundary[0, :]), 
                                   self.observation_space["goal"].high[:3]-self.observation_space["goal"].low[:3]) + self.observation_space["goal"].low[:3]
            norm_vel_xy = self.vel[i, :2] / MAX_LIN_VEL_XY
            norm_vel_z = self.vel[i, 2] / MAX_LIN_VEL_Z

            temp = {
                "state": np.hstack([norm_pos, self.quat[i, :], norm_vel_xy, norm_vel_z]),
                "goal": norm_goal
            }
            obs.append(temp)
        return obs

    def _computeReward(self):
        """Computes the current reward value(s).
        """
        # TODO: define reward
        return 0.

    def _computeDone(self):
        """Computes the current done value(s).
        """
        # TODO: define done
        return False

    def _computeInfo(self):
        """Computes the current done value(s).
        """
        # TODO: define info
        return {}

    def scale_action(self, action):
        """Scale action
        """
        scaled_action = np.multiply(np.divide(action-self.true_action_space.low, self.true_action_space.high-self.true_action_space.low), 
                             self.action_space.high-self.action_space.low) + self.action_space.low
        return scaled_action

    def _unscale_action(self, scaled_action):
        """Unscale action
        """
        action = np.multiply(np.divide(scaled_action-self.action_space.low, self.action_space.high-self.action_space.low), 
                             self.true_action_space.high-self.true_action_space.low) + self.true_action_space.low
        return action

    def _apply_control(self, action):
        """Pre-processes the action passed to `.step()` into motors' RPMs."""
        rpm = np.zeros((self.NUM_DRONES, 4))
        # TODO: to unscale action
        # no need for mutex as mutex is used outside in run_sim thread
        vel_cmd = self._unscale_action(action)
        for i in range(self.NUM_DRONES):
            temp, _, _ = self.ctrl[i].computeControl(control_timestep=self.CONTROL_TIMESTEP, 
                                                    cur_pos=self.pos[i, :],
                                                    cur_quat=self.quat[i, :],
                                                    cur_vel=self.vel[i, :],
                                                    cur_ang_vel=self.ang_v[i, :],
                                                    target_pos=self.pos[i, :], # same as the current position
                                                    target_rpy=np.array([0, 0, self.rpy[i, 2]]), # keep current yaw
                                                    target_vel=vel_cmd[i, :] # target the desired velocity vector
                                                    )
            rpm[i, :] = temp
        return rpm

    def getPyBulletClient(self):
        """Returns the PyBullet Client Id.

        Returns
        -------
        int:
            The PyBullet Client Id.
        """
        return self.CLIENT
    
    def getDroneIds(self):
        """Return the Drone Ids.

        Returns
        -------
        ndarray:
            (NUM_DRONES,)-shaped array of ints containing the drones' ids.
        """
        return self.DRONE_IDS
    
    def setPlanner(self, planner):
        """Set planner

        planner is expected to have a public method "plan" which generates planned action based on observation
        """
        self.planner = planner

    def setGoal(self, new_goal):
        self.curr_goal = new_goal

    def plan(self, obs):
        """Plan action based on current observation"""
        t0 = time.time()
        action = self.planner.plan(obs)
        t1 = time.time()
        self.curr_plan_time = t1 - t0
        return action

    def close(self):
        """Terminates the environment. """
        with self.bullet_sim_condition:
            self.stop_flag = True
            while self.running:
                self.bullet_sim_condition.wait()
            p.disconnect(physicsClientId=self.CLIENT)
    
    def reset(self):
        """Resets the environment.

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        """
        rospy.logwarn("[Resetting]")
        with self.bullet_sim_condition:
            self.is_resetting = True
            with self.bullet_sim_mutex:
                p.resetSimulation(physicsClientId=self.CLIENT)
                self._housekeeping()
                self._updateStateInformation()
                rospy.logwarn("[Housekeeping and resetting]")
            self.is_resetting = False
            self.bullet_sim_condition.notify_all()
        return self._computeObs()
    
    def step(self,
             action
             ):
        """Advances the environment by one plan step

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current epoisode is over, check the specific implementation of `_computeDone()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.
        """
        with self.bullet_sim_mutex:
            self.action = action
        time.sleep(self.PLAN_TIMESTEP - self.curr_plan_time)
        obs = self._computeObs()
        reward = self._computeReward()
        done = self._computeDone()
        info = self._computeInfo()
        return obs, reward, done, info


class DepthBulletDronesEnv(BulletDronesEnv):
    """The derived class which use onboard depth estimation as observation for collision avoidance"""
    