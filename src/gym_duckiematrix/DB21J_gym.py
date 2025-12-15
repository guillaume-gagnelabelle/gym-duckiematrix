"""
Gym mode version of DB21J environment.
This version runs faster by controlling simulation timing:
- Resets to last position with wheel speeds at start of step
- Sleeps inside step function
- Saves position and wheel speeds at end
- Sets wheel speeds to 0 at end

This allows non-real-time simulation for faster training.
"""

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
from duckietown.sdk.utils.loop_lane_position import (
    is_out_of_lane,
    compute_d,
    compute_d_signed,
    compute_theta,
    random_initial_position,
    perfect_initial_position,
    get_closest_tile,
)


DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
CURVED_TILES = {0, 2, 6, 8}


class DuckiematrixDB21JEnvGym(gym.Env):
    """
    Gym mode version of DuckiematrixDB21JEnv.
    Runs faster by controlling simulation timing instead of real-time.
    """
    def __init__(self, entity_name = "map_0/vehicle_0", out_of_road_penalty = -10.0, include_curve_flag: bool = False, step_duration: float = 0.1):
        """
        Args:
            entity_name: Entity name for the robot
            out_of_road_penalty: Penalty for going out of road
            include_curve_flag: Whether to include curve flag in observations
            step_duration: Duration of each step in seconds (default: 0.1)
        """
        self._shutdown = False
        self.include_curve_flag = include_curve_flag
        self.pose_reset_available = True
        self.step_duration = step_duration  # Duration to sleep in step function
        
        #create connection to the matrix engine
        self.robot: DB21J = DB21J("map_0/vehicle_0", simulated=True)
        self.initialize_sensors()
        self.action_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        # Observation: [signed_distance_from_center, theta] (+ optional curve flag)
        # signed_distance: negative = left side (white line), positive = right side (yellow line)
        if include_curve_flag:
            self.observation_space = spaces.Box(
                low=np.array([-0.3, -np.pi, 0.0], dtype=np.float32),
                high=np.array([0.3, np.pi, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=np.array([-0.3, -np.pi], dtype=np.float32),
                high=np.array([0.3, np.pi], dtype=np.float32),
                dtype=np.float32,
            )
        self.map = {"frames": None, "tiles": None, "tile_info": None}
        self.get_map()
        self.map_int = MapInterpreter(map=self.map)
        self.lp_cal = LanePositionCalculator(map_interpreter=self.map_int)
        self.out_of_road_penalty = out_of_road_penalty
        self.last_pose = None
        self.last_position = None  # Track last (x, y) position for progress calculation
        self.last_yaw = None  # Track last yaw angle for turning rate calculation
        self.last_forward_velocity = 0.0  # Track last forward velocity for smooth motion
        self._last_lateral_offset = None  # Track signed offset relative to lane centerline
        self.info : Dict = {}
        self._last_terminated_position = None  # Store position where termination occurred
        
        # Gym mode: track computation state position and wheel speeds for reset
        self._computation_state_position = None  # (x, y, yaw) position where action was computed for
        self._last_wheel_speeds = (0.0, 0.0)  # (left, right) wheel speeds from previous step (to preserve momentum)
        self._needs_initial_reset = True  # Flag to track if we need to reset position on first step

    def initialize_sensors(self):
        #self.robot.camera.start()
        self.robot.motors.start()
        self.robot.map_frames.start()
        self.robot.map_tiles.start()
        self.robot.map_tile_info.start()
        self.robot.pose.start()
        self.robot.reset_flag.start()
        try:
            self.robot.pose_reset.start()
        except Exception as exc:  # pose_reset endpoint may be missing in some setups
            print(f"[WARN] pose_reset not available: {exc}")
            self.pose_reset_available = False
        
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
        
    def reward_fn(self, d, theta, action, delta_t, x, y, yaw):
        """
        Simplified reward:
        -1.0 if the bot leaves the lane or heading diverges too much.
        Otherwise 0.1 * (speed_aligned_with_lane - distance_from_center).
        """
        # Terminate-style penalty if clearly out of bounds or wildly off-heading.
        max_lane_offset = 0.585 / 2
        if d < 0 or d > max_lane_offset or abs(theta) > math.pi / 2:
            return -1.0

        if delta_t is None or delta_t <= 0 or self.last_position is None:
            return 0.0

        dx = x - self.last_position[0]
        dy = y - self.last_position[1]
        displacement = math.sqrt(dx**2 + dy**2)

        safe_delta_t = max(delta_t, 1e-4)
        linear_speed = displacement / safe_delta_t

        alignment = math.cos(theta)  # 1 when aligned, 0 when perpendicular
        aligned_speed = linear_speed * alignment

        # Forward-focused shaping; backward motion already limited by action clipping
        reward = 0.1 * (aligned_speed - d)
        self.last_forward_velocity = aligned_speed
        return reward

    def step(self, actions : Tuple) -> Tuple:
        """
        Gym mode step (simulates computation delay - real-time RL):
        Flow exactly as specified:
        1. Reset to computation_state_position (teleportation to S₀)
        2. Capture pose and save as last_pose (state action was computed for)
        3. Restore previous wheel speeds (preserve momentum from before stopping)
        4. Apply new wheel speeds from action
        5. Sleep for step_duration (simulating computation delay + execution)
        6. Capture pose immediately after sleep (frozen state - this is where bot will be teleported next time)
        7. Compute obs from frozen pose
        8. Compute reward from frozen pose
        9. Save frozen pose and wheel speeds (for next step's reset and momentum preservation)
        10. Stop motors and return
        
        Key points:
        - Pose is captured IMMEDIATELY after sleep, before any computation
        - Wheel speeds are preserved: robot continues with previous velocities after teleport
        - This ensures the "frozen" state is independent of how long it takes to compute the next action
        - The step_duration simulates the delay between computing and executing the action
        """
        step_start_time = time.perf_counter()
        
        # Step 1: Reset to computation_state_position (where action was computed for)
        # This simulates: "Action was computed for this state, now we apply it"
        # On first step after reset(), we use the reset position
        reset_position = self._computation_state_position
        if reset_position is None and self._needs_initial_reset:
            # First step after reset - use the position from reset()
            # We'll get this from the last_pose or from a saved position
            if self.last_pose is not None:
                reset_x = self.last_pose["position"]["x"]
                reset_y = self.last_pose["position"]["y"]
                reset_yaw = compute_yaw(self.last_pose)
                reset_position = (reset_x, reset_y, reset_yaw)
        
        if reset_position is not None:
            x, y, yaw = reset_position
            # Check if the position is valid (within lane bounds)
            if not is_out_of_lane(x, y):
                qw = np.cos(yaw / 2.0)
                qz = np.sin(yaw / 2.0)
                header = Header(timestamp=float(time.time()))
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
                if self.pose_reset_available:
                    try:
                        # Restore wheel speeds BEFORE teleport so robot continues moving immediately after
                        # This prevents the robot from stopping between steps
                        if self._last_wheel_speeds is not None:
                            prev_wl, prev_wr = self._last_wheel_speeds
                            if prev_wl != 0.0 or prev_wr != 0.0:
                                # Set wheel speeds before teleport so they're active immediately after
                                self.robot.motors.set_pwm(left=prev_wl, right=prev_wr)
                        
                        self.robot.pose_reset.set_pose(teleport)
                        # CRITICAL: pose_reset.set_pose() likely resets motor state in physics engine
                        # Re-set wheel speeds IMMEDIATELY after teleport to prevent stopping
                        if self._last_wheel_speeds is not None:
                            prev_wl, prev_wr = self._last_wheel_speeds
                            if prev_wl != 0.0 or prev_wr != 0.0:
                                # Set wheel speeds immediately after teleport (teleport may have cleared them)
                                self.robot.motors.set_pwm(left=prev_wl, right=prev_wr)
                        
                        # Small wait for teleport to take effect
                        time.sleep(0.05)
                        # Verify reset worked by checking pose
                        verify_pose = self.robot.pose.capture()
                        if verify_pose is not None:
                            reset_x = verify_pose["position"]["x"]
                            reset_y = verify_pose["position"]["y"]
                            # Check if reset was successful (within 0.1m tolerance)
                            dist = math.sqrt((reset_x - x)**2 + (reset_y - y)**2)
                            if dist > 0.1:
                                # Reset might not have worked, try once more
                                self.robot.pose_reset.set_pose(teleport)
                                # Re-set wheel speeds after retry (teleport clears them)
                                if self._last_wheel_speeds is not None:
                                    prev_wl, prev_wr = self._last_wheel_speeds
                                    if prev_wl != 0.0 or prev_wr != 0.0:
                                        self.robot.motors.set_pwm(left=prev_wl, right=prev_wr)
                                time.sleep(0.05)
                    except Exception as exc:
                        print(f"[WARN] pose_reset.set_pose failed: {exc}")
        
        # Step 2: Capture pose and save as last_pose (this is the state action was computed for)
        computation_pose = self.robot.pose.capture()
        if computation_pose is None:
            # Wait a bit and try again
            time.sleep(0.01)
            computation_pose = self.robot.pose.capture()
        if computation_pose is not None:
            self.last_pose = computation_pose
        
        # Step 3: Ensure wheel speeds are set (they should already be set before teleport above)
        # But if they weren't set (e.g., first step after reset), set them now
        if self._last_wheel_speeds is not None:
            prev_wl, prev_wr = self._last_wheel_speeds
            if prev_wl != 0.0 or prev_wr != 0.0:
                # Make sure motors are set (in case teleport cleared them)
                self.robot.motors.set_pwm(left=prev_wl, right=prev_wr)
                time.sleep(0.02)  # Small delay for motors to engage
            # If prev_wl == 0.0 and prev_wr == 0.0, we skip (fresh start after hard reset)
        
        # Step 4: Apply new action (computed for S₀, now we apply it)
        # The robot continues with previous velocities, then we apply the new action
        safe_actions = np.clip(actions, 0.0, 1.0)
        wl = safe_actions[0] * 0.4
        wr = safe_actions[1] * 0.4
        
        # Set motors to new action values
        self.robot.motors.set_pwm(left=wl, right=wr)
        
        # Small delay for motors to engage
        time.sleep(0.05)
        
        # Step 5: Sleep for step_duration (simulating computation delay + execution)
        # During this time, the robot moves with the new action
        # Sleep in chunks and re-set motors periodically to ensure they stay on
        overhead_before_sleep = time.perf_counter() - step_start_time
        estimated_post_overhead = 0.015  # Estimate for pose capture + computation
        
        # For step_duration=0.0, use a minimal time to allow robot to move
        # Otherwise it would capture pose immediately and keep resetting to same position
        effective_step_duration = max(self.step_duration, 0.01) if self.step_duration == 0.0 else self.step_duration
        
        remaining_time = effective_step_duration - overhead_before_sleep - estimated_post_overhead
        
        if remaining_time > 0:
            # Sleep in chunks, re-setting motors periodically to ensure they stay on
            chunk_size = 0.1  # Sleep in 100ms chunks
            elapsed = 0.0
            while elapsed < remaining_time:
                chunk = min(chunk_size, remaining_time - elapsed)
                time.sleep(chunk)
                elapsed += chunk
                # Re-set motors periodically to ensure they stay on
                if elapsed < remaining_time:  # Don't re-set on last chunk
                    self.robot.motors.set_pwm(left=wl, right=wr)
        elif self.step_duration == 0.0:
            # Even for 0.0 delay, give robot minimal time to move
            # This prevents continuous resetting to the same position
            time.sleep(0.01)
            self.robot.motors.set_pwm(left=wl, right=wr)  # Ensure motors are on
        
        # Step 5: Capture final pose (after robot has moved with action)
        # This is where the robot is now, and where it will be teleported to next time
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
                obs_vals = [0.0, 0.0]
                if self.include_curve_flag:
                    obs_vals.append(0.0)
                obs = np.array(obs_vals, dtype=np.float32)
                reward = 0.0
                terminated = False
                truncated = False
                self.info = {"pose": None, "tile": None, "is_curve_tile": None}
                info = self._get_info()
                # Set wheel speeds to 0 before returning
                self.robot.motors.set_pwm(left=0.0, right=0.0)
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
        if d_signed == -1:  # Out of lane sentinel
            d_signed_clamped = 0.3
        else:
            d_signed_clamped = max(-0.3, min(0.3, d_signed))

        tile_id = get_closest_tile(x, y)
        is_curve_tile = 1.0 if tile_id in CURVED_TILES else 0.0

        obs_vals = [d_signed_clamped, theta]
        if self.include_curve_flag:
            obs_vals.append(is_curve_tile)
        obs = np.array(obs_vals, dtype=np.float32)

        # Only terminate if actually out of lane bounds
        terminated = is_out_of_lane(x, y)
        
        # Store the position where termination occurred
        if terminated:
            self._last_terminated_position = (x, y, yaw)
            self.info["terminated_position"] = (x, y, yaw)
        
        truncated = False
        # Step 5: Compute reward
        reward = self.reward_fn(d, theta, actions, delta_t, x, y, yaw)

        # Update last position and yaw for next step's calculations
        self.last_position = (x, y)
        self.last_yaw = yaw
        self.last_pose = pose

        # Step 6: Save position and wheel speeds for next step
        # This is the "frozen" pose captured immediately after sleep
        # This is where the NEXT action will be computed for (and where we'll teleport to)
        # IMPORTANT: Save wheel speeds so we can restore them after teleport
        # Only save if we didn't terminate (don't save out-of-bounds positions)
        if not terminated:
            self._computation_state_position = (x, y, yaw)
            self._last_wheel_speeds = (wl, wr)  # Save current wheel speeds for next step
            # Don't stop motors - robot can continue moving, pose is already captured
        else:
            # Hard reset: Clear the saved position and stop motors
            # This is a termination (out of bounds), so we need a proper reset
            self._computation_state_position = None
            self._last_wheel_speeds = (0.0, 0.0)
            self._needs_initial_reset = True  # Will need reset on next episode
            # Stop motors for hard reset (termination)
            self.robot.motors.set_pwm(left=0.0, right=0.0)

        self.info = {"pose": pose, "tile": tile_id, "is_curve_tile": bool(is_curve_tile)}
        info = self._get_info()
        
        # Note: Motors are NOT stopped for normal steps
        # The robot can continue moving - the pose is already captured and saved
        # Motors will be stopped only on termination (hard reset) above
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
        # Stop motors first
        # self.robot.motors.set_pwm(left=0.0, right=0.0)
        self._last_wheel_speeds = (0.0, 0.0)  # Reset wheel speeds on environment reset
        
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
        self.last_position = (x, y)  # Initialize last_position to reset position
        self.last_yaw = yaw  # Initialize last_yaw for turning rate calculation
        self.last_forward_velocity = 0.0  # Reset forward velocity tracking
        self._last_lateral_offset = None
        
        # Reset gym mode tracking
        self._needs_initial_reset = True  # Will reset position on first step after this reset()
        
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
        
        # Stop motors FIRST before resetting pose - important for clean reset
        # (Already done above, but keeping comment for clarity)
        
        if self.pose_reset_available:
            try:
                self.robot.pose_reset.set_pose(teleport)
            except Exception as exc:
                print(f"[WARN] pose_reset.set_pose failed (continuing without teleport): {exc}")
                self.pose_reset_available = False
        
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
            # Set computation_state_position - this is where the first action will be computed for
            self._computation_state_position = (
                new_pose["position"]["x"],
                new_pose["position"]["y"],
                compute_yaw(new_pose)
            )
        else:
            # Fallback to requested position if we couldn't capture pose
            self._computation_state_position = (x, y, yaw)

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
        obs_x, obs_y = self.last_position if self.last_position is not None else (x, y)
        obs_yaw = self.last_yaw if self.last_yaw is not None else yaw
        tile_id = get_closest_tile(obs_x, obs_y)
        # Use compute_d_signed instead of compute_d to match your friend's version
        obs_vals = [compute_d_signed(obs_x, obs_y), compute_theta(obs_x, obs_y, obs_yaw)]
        if self.include_curve_flag:
            obs_vals.append(1.0 if tile_id in CURVED_TILES else 0.0)
        obs = np.array(obs_vals, dtype=np.float32)
        self.info = {"pose": self.last_pose, "tile": tile_id, "is_curve_tile": tile_id in CURVED_TILES}
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

