from typing import Tuple, Dict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import math
from duckietown_messages.geometry_3d import Transformation, Position, Quaternion
from duckietown_messages.standard import Header
from duckietown.sdk.robots.duckiebot import DB21J
from duckietown.sdk.utils.lane_position import MapInterpreter, LanePositionCalculator
from .utils import quaternion_to_euler, compute_yaw
from duckietown.sdk.utils.loop_lane_position import *

DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480


class DuckiematrixDB21JEnv(gym.Env):
    def __init__(self, entity_name = "map_0/vehicle_0", out_of_road_penalty = -1.0):

        self._shutdown = False
        #create connection to the matrix engine
        self.robot: DB21J = DB21J("map_0/vehicle_0", simulated=True)
        self.initialize_sensors()
        #self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        # Observation: [signed_distance_from_center, theta]
        # signed_distance: negative = left side (white line), positive = right side (yellow line)
        #self.observation_space = spaces.Box(low=np.array([-0.3, -np.pi]), high=np.array([0.3, np.pi]), dtype=np.float32)

        self.map = {"frames": None, "tiles": None, "tile_info": None}
        self.get_map()
        self.map_int = MapInterpreter(map=self.map)
        self.lp_cal = LanePositionCalculator(map_interpreter=self.map_int)
        self.out_of_road_penalty = out_of_road_penalty
        self.last_pose = None
        self.last_position = None  # Track last (x, y) position for progress calculation
        self.last_yaw = None  # Track last yaw angle for turning rate calculation
        self.last_forward_velocity = 0.0  # Track last forward velocity for smooth motion
        self.info : Dict = {}
        self._last_terminated_position = None  # Store position where termination occurred
        self.distance = 0.0

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
        
        
    def reward_fn(self, d, theta, action):

        dist_from_center = abs(d)
        
        forward = np.sqrt(action[0] ** 2 + action[1] ** 2)
        align = math.cos(theta)
        
        return 0.1 * (forward * align - dist_from_center)


    def step(self, actions : Tuple) -> Tuple:
        # TODO: this is a hack to simulate rad/s to PWM conversion
        wl = actions[0]*0.4
        wr = actions[1]*0.4

        self.robot.motors.set_pwm(left=wl, right=wr)

        pose = self.robot.pose.capture()
        
        # Wait for pose if None (simulator might not have updated yet)
        max_wait = 10
        wait_count = 0
        truncated = False
        while pose is None and wait_count < max_wait:
            time.sleep(0.01)
            pose = self.robot.pose.capture()
            wait_count += 1
        
        # If still None, use last pose or return default
        if pose is None:
            if self.last_pose is not None:
                pose = self.last_pose
            else:
                # Return default observation if no pose available
                obs = np.array([0.0, 0.0], dtype=np.float32)
                reward = 0.0   # TODO: CHANGE CODE SO THAT Q_TABLE DOESN'T GET UPDATED WHEN THIS HAPPENS. SHOULD USE TRUNCATED FOR THAT - DONE =)
                terminated = False
                truncated = True
                self.distance = 0
                self.info = {"pose": None}
                self.info["distance"]= self.distance
                info = self._get_info()
                return obs, reward, terminated, truncated, info

        x, y, yaw = pose["position"]["x"], pose["position"]["y"], compute_yaw(pose)
        last_x, last_y = self.last_pose["position"]["x"], self.last_pose["position"]["y"]
        obs = np.array([compute_d_signed(x, y), compute_theta(x, y, yaw), np.int32(in_curve(x, y))], dtype=np.float32)

        terminated = is_out_of_lane(x, y) or abs(obs[1]) > math.pi / 2

        reward = self.reward_fn(obs[0], obs[1], actions) if not terminated else self.out_of_road_penalty

        self.distance = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5
        self.last_pose = pose

        self.info = {"pose": pose}
        self.info["distance"] = self.distance

        info = self._get_info()
        return obs, reward, terminated, truncated, info
        

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
        self.last_position = (x, y)  # Initialize last_position to reset position
        self.last_yaw = yaw  # Initialize last_yaw for turning rate calculation
        self.last_forward_velocity = 0.0  # Reset forward velocity tracking
        self.distance = 0.0
        # send teleport command to the simulator (if supported)
        # clear any stale pose so we can wait for a fresh post-reset reading
        self.robot.pose.capture()
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
        
        self.robot.motors.set_pwm(left=0, right=0)
        self.robot.pose_reset.set_pose(teleport)
        # try to grab a fresh pose after requesting the reset; prefer one close to target
        t_start = time.time()
        new_pose = None
        while time.time() - t_start < 0.5:
            candidate = self.robot.pose.capture(block=True, timeout=0.1)
            if candidate is None:
                continue
            cx, cy = candidate["position"]["x"], candidate["position"]["y"]
            cyaw = compute_yaw(candidate)
            close_pos = abs(cx - x) < 0.05 and abs(cy - y) < 0.05
            yaw_diff = (cyaw - yaw + math.pi) % (2 * math.pi) - math.pi
            close_yaw = abs(yaw_diff) < 0.05
            if close_pos and close_yaw:
                new_pose = candidate
                break
        # fall back to any captured pose if no close match found
        if new_pose is None:
            new_pose = candidate if "candidate" in locals() else None
        if new_pose is not None:
            self.last_pose = new_pose
            # Update last_position with actual reset position
            self.last_position = (new_pose["position"]["x"], new_pose["position"]["y"])

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
        obs = np.array([compute_d_signed(x, y), compute_theta(x, y, yaw), in_curve(x, y)], dtype=np.float32)
        self.info = {"pose": self.last_pose}
        self.info["distance"] = self.distance
        info = self._get_info()
        return obs, info


    def _get_info(self) -> Dict:
        """Get the info for each robot in the environment

        Returns:
            info (Dict): A info dictionary with info for each robot
        """
        return self.info
