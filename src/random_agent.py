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
info = env.reset(position=desired_reset)
sleep(0.1)
i = 0
for step in range(5000):

    action = [1, 1]
    (d, theta), reward, terminated, info = env.step(action)
    sleep(0.1)

    if info is not None:
        print_step_info(step, action, reward, terminated, d, theta, info)
    #p = env.robot.pose.capture() # for some reason this creates bugs, probably a delay issue
    
    if info is not None:
        if terminated: # or is_out_of_bounds(info.get("pose")):
            print("$$$$$$$$$$$$$$$$$$$$ ENVIRONMENT RESETTING $$$$$$$$$$$$$$$$$$$$")
            info = env.reset(position=desired_reset)
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
