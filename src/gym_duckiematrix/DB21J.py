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
        # Observation: [signed_distance_from_center, theta]
        # signed_distance: negative = left side (white line), positive = right side (yellow line)
        self.observation_space = spaces.Box(low=np.array([-0.3, -np.pi]), high=np.array([0.3, np.pi]), dtype=np.float32)
        #self.observation_space = spaces.Box(
        #    low=0, high=255, shape=(DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, 3), dtype=np.uint8
        #)
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

        """
        Reward function that encourages smooth forward motion and discourages turning toward yellow line.
        
        Args:
            d: Distance from lane center (absolute)
            theta: Angle between desired heading and actual heading
            action: Action taken
            delta_t: Time delta
            x, y: Current position
            yaw: Current yaw angle
        
        # Large penalty for going out of bounds
        if d > 0.585 / 2 or abs(theta) > math.pi / 2:
            return self.out_of_road_penalty
        
        # If no valid time delta or no previous position, give zero reward (neutral)
        if delta_t is None or delta_t <= 0 or self.last_position is None:
            return 0.0
        
        # Calculate displacement from last position
        dx = x - self.last_position[0]
        dy = y - self.last_position[1]
        displacement = math.sqrt(dx**2 + dy**2)
        
        # If not moving, give small negative reward (encourages movement)
        if displacement < 1e-6:
            return -0.1
        
        # Compute forward progress: displacement projected onto desired direction
        forward_progress = displacement * math.cos(theta)
        
        # Compute forward velocity (m/s) - encourages smooth, flowing motion
        forward_velocity = 0.0
        if delta_t > 0:
            forward_velocity = forward_progress / delta_t
        
        # 1. REWARD: Forward progress (encourages advancing through lane)
        if forward_progress > 0:
            forward_reward = 15.0 * forward_progress  # Strong reward for forward distance
        else:
            forward_reward = 30.0 * forward_progress  # Heavy penalty for backward (double magnitude)
        
        # 2. REWARD: Forward velocity (encourages smooth, flowing movement)
        # Reward maintaining good forward speed (0.1-0.5 m/s is good)
        velocity_reward = 0.0
        if forward_velocity > 0.05:  # Only reward if moving forward
            # Reward increases with velocity up to a point, then plateaus
            velocity_reward = 3.0 * min(forward_velocity, 0.3)  # Max reward at 0.3 m/s
        
        # 3. PENALTY: Turning too much (discourages excessive turning)
        turning_penalty = 0.0
        if self.last_yaw is not None and delta_t > 0:
            # Compute angular velocity (rad/s)
            yaw_diff = yaw - self.last_yaw
            # Normalize to [-pi, pi]
            yaw_diff = math.atan2(math.sin(yaw_diff), math.cos(yaw_diff))
            angular_velocity = abs(yaw_diff) / delta_t
            
            # Penalize high angular velocity (turning too fast)
            # Angular velocity > 1.0 rad/s is considered excessive turning
            if angular_velocity > 0.5:
                turning_penalty = 2.0 * (angular_velocity - 0.5)  # Penalty increases with turning rate
        
        # 4. PENALTY: Approaching yellow line (strongly discourages going toward yellow line)
        d_signed = compute_d_signed(x, y)
        yellow_line_penalty = 0.0
        if d_signed > 0:  # On the right side (toward yellow line)
            # Very strong penalty that increases quadratically as we approach yellow line
            # At d_signed = 0.0 (center): penalty = 0
            # At d_signed = 0.1 (close to yellow): penalty = 10.0
            yellow_line_penalty = 10.0 * (d_signed ** 2)  # Quadratic penalty
        
        # 5. PENALTY: Being off-center (encourages staying in lane center, but less on white line side)
        if d_signed < 0:  # On the left side (toward white line) - safer
            lane_penalty = 0.2 * abs(d)  # Small penalty
        else:  # On the right side (toward yellow line) - dangerous
            lane_penalty = 1.0 * abs(d)  # Larger penalty
        
        # 6. PENALTY: Heading error (encourages alignment with lane direction)
        # Only penalize significant misalignment to allow for small corrections
        heading_penalty = 0.5 * max(0, abs(theta) - 0.1)  # Only penalize if > 0.1 rad (~5.7°)
        
        # Total reward: rewards - penalties
        reward = forward_reward + velocity_reward - turning_penalty - yellow_line_penalty - lane_penalty - heading_penalty
        
        # Update tracking variables
        self.last_forward_velocity = forward_velocity
        
        return reward
        """

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
                self.info = {"pose": None}
                info = self._get_info()
                return obs, reward, terminated, truncated, info

        x, y, yaw = pose["position"]["x"], pose["position"]["y"], compute_yaw(pose)
        obs = np.array([compute_d_signed(x, y), compute_theta(x, y, yaw), np.int32(in_curve(x, y))], dtype=np.float32)

        terminated = is_out_of_lane(x, y) or abs(obs[1]) > math.pi / 2

        reward = self.reward_fn(obs[0], obs[1], actions) if not terminated else self.out_of_road_penalty

        self.last_pose = pose

        self.info = {"pose": pose}
        info = self._get_info()
        return obs, reward, terminated, truncated, info
        
        """
        # TODO: this is a hack to simulate rad/s to PWM conversion
        wl = actions[0]*0.4
        wr = actions[1]*0.4

        self.robot.motors.set_pwm(left=wl, right=wr)
        #bgr = self.robot.camera.capture()
        
        #if bgr is None:
        #    print("got no image.. skipping")
        #    return None, None, None, None, None, None
        
        pose = self.robot.pose.capture()
        
        # Wait for pose if None (simulator might not have updated yet)
        max_wait = 10
        wait_count = 0
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
                reward = 0.0
                terminated = False
                truncated = False
                self.info = {"pose": None}
                info = self._get_info()
                return obs, reward, terminated, truncated, info
        
        delta_t = None
        if self.last_pose is not None and pose is not None:
            delta_t = float(pose["header"]["timestamp"]) - float(self.last_pose["header"]["timestamp"])

        x, y, yaw = pose["position"]["x"], pose["position"]["y"], compute_yaw(pose)
        d_signed = compute_d_signed(x, y)
        theta = compute_theta(x, y, yaw)
        d = abs(d_signed) if d_signed >= 0 else compute_d(x, y)  # Fallback to abs if signed fails
        # Clamp d_signed to observation space bounds [-0.3, 0.3]
        # If out of lane (d_signed == -1), use a large value to indicate out of bounds
        if d_signed < 0:  # Out of lane
            d_signed_clamped = 0.3  # Use max value to indicate problem
        else:
            d_signed_clamped = max(-0.3, min(0.3, d_signed))
        obs = np.array([d_signed_clamped, theta], dtype=np.float32)

        # Only terminate if actually out of lane bounds
        # Don't terminate on theta alone - the reward function already penalizes high theta
        # This allows the agent to make corrections and recover from mistakes
        terminated = is_out_of_lane(x, y)
        
        # Store the position where termination occurred
        if terminated:
            self._last_terminated_position = (x, y, yaw)
            # Also store in info for access from agent
            self.info["terminated_position"] = (x, y, yaw)
        
        truncated = False
        # Pass absolute distance d to reward function (for termination check)
        reward = self.reward_fn(d, theta, actions, delta_t, x, y, yaw)

        # Update last position and yaw for next step's calculations
        self.last_position = (x, y)
        self.last_yaw = yaw
        self.last_pose = pose

        #rgb = bgr[:, :, [2,1,0]]
        #self.window.set_data(rgb)
        #self.fig.canvas.draw_idle()
        #self.fig.canvas.start_event_loop(0.00001)

        self.info = {"pose": pose}
        info = self._get_info()
        return obs, reward, terminated, truncated, info
        #return rgb, reward, terminated, d, theta, info
        """
        
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
        #obs = self.robot.camera.capture()
        #print("x = %.2f, y = %.2f" %(x, y))
        obs = np.array([compute_d_signed(x, y), compute_theta(x, y, yaw), in_curve(x, y)], dtype=np.float32)
        #print("d = %.2f" % obs[0])
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
