from typing import Tuple, Dict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from duckietown_messages.geometry_3d import Transformation, Position, Quaternion
from duckietown_messages.standard import Header
from duckietown.sdk.robots.duckiebot import DB21J
from duckietown.sdk.utils.lane_position import MapInterpreter, LanePositionCalculator
from .utils import quaternion_to_euler, compute_yaw
from duckietown.sdk.utils.loop_lane_position import is_out_of_lane, compute_d, compute_theta, random_initial_position, perfect_initial_position, get_closest_tile
import math


DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480


class DuckiematrixDB21JEnv(gym.Env):
    def __init__(self, entity_name = "map_0/vehicle_0", out_of_road_penalty = -10.0):
        #import matplotlib.pyplot as plt
        # create matplot window
        #self.window = plt.imshow(np.zeros((DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, 3)))
        #plt.axis("off")
        #self.fig = plt.figure(1)
        #plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        #plt.pause(0.01)

        self._shutdown = False
        #create connection to the matrix engine
        self.robot: DB21J = DB21J("map_0/vehicle_0", simulated=True)
        self.initialize_sensors()
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, -np.pi]), high=np.array([0.5, np.pi]), dtype=np.float32)
        #self.observation_space = spaces.Box(
        #    low=0, high=255, shape=(DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, 3), dtype=np.uint8
        #)
        self.map = {"frames": None, "tiles": None, "tile_info": None}
        self.get_map()
        self.map_int = MapInterpreter(map=self.map)
        self.lp_cal = LanePositionCalculator(map_interpreter=self.map_int)
        self.out_of_road_penalty = out_of_road_penalty
        self.last_pose = None
        self.info : Dict = {}
        self._last_terminated_position = None  # Store position where termination occurred

    def initialize_sensors(self):
        #self.robot.camera.start()
        self.robot.motors.start()
        self.robot.map_frames.start()
        self.robot.map_tiles.start()
        self.robot.map_tile_info.start()
        self.robot.pose.start()
        self.robot.reset_flag.start()
        self.robot.pose_reset.start()
        
    def get_map(self):
        while True:
            if self.check_map():
                break
            if self.map["frames"] is None:
                self.map["frames"] = self.robot.map_frames.capture()
            elif self.map["tiles"] is None:
                self.map["tiles"] = self.robot.map_tiles.capture()
            elif self.map["tile_info"] is None:
                self.map["tile_info"] = self.robot.map_tile_info.capture()
            
    def check_map(self):
        is_map = True
        for key in self.map.keys():
            if self.map[key] is None:
                is_map = False 
        return is_map               
        
        
    def reward_fn(self, d, theta, action, delta_t):

        if d > 0.585 / 2 or abs(theta) > math.pi / 2: # Assuming d and theta are computed correctly, then True implies the robots is out of lane or wrong direction
            return self.out_of_road_penalty
        
        if delta_t is None or delta_t <= 0:
            # If we are out of lane or cannot infer a sensible speed, penalize.
            return 0.0

        action_norm = float(np.linalg.norm(action))
        speed_proxy = action_norm / delta_t

        # Encourage forward alignment, discourage lateral/heading error.
        alignment = max(0.0, math.cos(theta))
        lane_penalty = 4.0 * abs(d) + 1.0 * abs(theta)

        return speed_proxy * alignment - lane_penalty

    def step(self, actions : Tuple) -> Tuple:
        # TODO: this is a hack to simulate rad/s to PWM conversion
        wl = actions[0]*0.4
        wr = actions[1]*0.4

        self.robot.motors.set_pwm(left=wl, right=wr)
        #bgr = self.robot.camera.capture()
        
        #if bgr is None:
        #    print("got no image.. skipping")
        #    return None, None, None, None, None, None
        
        pose = self.robot.pose.capture()
        delta_t = None
        if self.last_pose is not None and pose is not None:
            delta_t = float(pose["header"]["timestamp"]) - float(self.last_pose["header"]["timestamp"])

        x, y, yaw = pose["position"]["x"], pose["position"]["y"], compute_yaw(pose)
        obs = np.array([compute_d(x, y), compute_theta(x, y, yaw)], dtype=np.float32)

        terminated = is_out_of_lane(x, y) or abs(obs[1]) > math.pi / 2
        
        # Store the position where termination occurred
        if terminated:
            self._last_terminated_position = (x, y, yaw)
            # Also store in info for access from agent
            self.info["terminated_position"] = (x, y, yaw)
        
        truncated = False
        reward = self.reward_fn(obs[0], obs[1], actions, delta_t)

        self.last_pose = pose

        #rgb = bgr[:, :, [2,1,0]]
        #self.window.set_data(rgb)
        #self.fig.canvas.draw_idle()
        #self.fig.canvas.start_event_loop(0.00001)

        self.info = {"pose": pose}
        info = self._get_info()
        return obs, reward, terminated, truncated, info
        #return rgb, reward, terminated, d, theta, info

    def reset(self, position: Tuple[float, float, float] | None = None, curve_prob: float = 0.5, perfect: bool = True, tile: int | None = None):
        """
        Reset the environment.
        
        Args:
            position: Specific (x, y, yaw) position to reset to. If None, uses perfect or random position.
            curve_prob: Probability of choosing a curved tile (only used if position is None and tile is None).
            perfect: If True and position is None, uses perfect_initial_position (exact center, perfect heading).
                    If False, uses random_initial_position (with jitter).
            tile: Specific tile number (0-8) to reset to. If provided, generates perfect position in that tile.
        """
        # If no position provided, use perfect position by default
        if position is None:
            if tile is not None:
                # Generate perfect position in the specified tile
                position = perfect_initial_position(tile=tile, position_along_tile=0.5)
            elif perfect:
                position = perfect_initial_position(curve_prob=curve_prob)
            else:
                position = random_initial_position(curve_prob)

        x, y, yaw = position
        # construct a minimal pose dict similar to the one produced by
        # the pose driver. We keep z/roll/pitch zero and use a simple
        # quaternion for yaw-only rotation.
        qw = np.cos(yaw / 2.0)
        qz = np.sin(yaw / 2.0)
        synthetic_pose = {
            "header": {"timestamp": float(0)},
            "position": {"x": float(x), "y": float(y), "z": 0.0},
            "rotation": {"w": float(qw), "x": 0.0, "y": 0.0, "z": float(qz)},
        }
        # set last_pose so reward_fn and other internals start from here
        self.last_pose = synthetic_pose
        # send teleport command to the simulator (if supported)
        header = Header(timestamp=float(0))
        position_msg = Position(header=header, x=float(x), y=float(y), z=0.0)
        rotation_msg = Quaternion(header=header, w=float(qw), x=0.0, y=0.0, z=float(qz))
        target = getattr(self.robot, "_name", "") or ""
        teleport = Transformation(
            header=header,
            source="",
            target=target,
            position=position_msg,
            rotation=rotation_msg,
        )
        self.robot.pose_reset.set_pose(teleport)
        # try to grab a fresh pose after requesting the reset
        new_pose = self.robot.pose.capture(block=True, timeout=0.5)
        if new_pose is not None:
            self.last_pose = new_pose

        # Log the starting position to help users verify the reset
        #print(f"Initial robot position: (x={round(x,2)}, y={round(y,2)}, yaw={round(yaw, 2)})")

        # perform the environment reset sequence used previously
        while True:
            # if last_pose already provided above, avoid overwriting it with
            # a None capture; otherwise capture the real pose from the robot.
            if self.last_pose is None:
                self.last_pose = self.robot.pose.capture()
            else:
                # try to update it with a new capture if available
                newer_pose = self.robot.pose.capture()
                if newer_pose is not None:
                    self.last_pose = newer_pose
            if self.last_pose is not None:
                break


        # inform the engine to reset the robot state (engine may or may not
        # act on this depending on its capabilities)
        self.robot.reset_flag.set_reset(True)
        #obs = self.robot.camera.capture()
        obs = np.array([compute_d(x, y), compute_theta(x, y, yaw)], dtype=np.float32)
        self.info = {"pose": self.last_pose}
        info = self._get_info()
        return obs, info

    def _get_reward(self) -> float:
        #TODO
        return 0.0

    def _get_info(self) -> Dict:
        """Get the info for each robot in the environment

        Returns:
            info (Dict): A info dictionary with info for each robot
        """
        return self.info