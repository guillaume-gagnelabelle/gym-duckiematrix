from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv
from time import sleep
import math
from gym_duckiematrix.utils import *
from duckietown.sdk.utils.loop_lane_position import *

# TODO
# Find the exact bounds of the map programmatically and put it in a file
# implement a better reset mechanism that doesn't rely on repeated attempts
# implement clean rl for action selection
# time how long each action takes
# make observation only the distance to center of lane and yaw


env = DuckiematrixDB21JEnv(entity_name="map_0/vehicle_0")

tile_size = 0.585
lane_width = 0.585 / 2

start_x = lane_width / 2
start_y = lane_width/2
start_yaw = 0.

desired_reset = (start_x, start_y, start_yaw)
_, info = env.reset(position=desired_reset)
sleep(0.1)
i = 0
last_pose = None

for step in range(5000):

    action = [1, 1]
    (d, theta), reward, terminated, truncated, info = env.step(action)

    sleep(0.1)

    print_step_info(step, action, reward, terminated, d, theta, info)
    
    # Track the last pose before termination
    if not terminated:
        last_pose = info.get("pose")
    
    if terminated:
        print("$$$$$$$$$$$$$$$$$$$$ ENVIRONMENT RESETTING $$$$$$$$$$$$$$$$$$$$")
        
        # Get the position where termination occurred
        terminated_pos = info.get("terminated_position")
        if terminated_pos is None and last_pose is not None:
            # Fallback: use last pose if terminated_position not available
            terminated_pos = (last_pose["position"]["x"], last_pose["position"]["y"], 0.0)
        
        # Determine which tile to reset to
        reset_tile = None
        if terminated_pos is not None:
            x, y, _ = terminated_pos
            reset_tile = get_closest_tile(x, y)
            print(f"[RESET] Robot went out of bounds at ({x:.3f}, {y:.3f}), resetting to tile {reset_tile}")
        
        # Reset to perfect position in that tile
        _, info = env.reset(tile=reset_tile)
        sleep(0.1)
            
        start_yaw = (i * math.pi / 8)
        if i % 16 == 0:
            i = 0
            start_yaw = 0.
            start_x += lane_width
            if start_x > 3 * tile_size:
                start_x = lane_width / 2
                start_y += lane_width/2
                if start_y > 3 * tile_size:
                    start_y = lane_width / 2
        desired_reset = start_x, start_y, start_yaw
        i += 1

env.robot.camera.stop()
env.robot.motors.stop()
