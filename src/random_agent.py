from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv
from time import sleep
import math

env = DuckiematrixDB21JEnv(entity_name="map_0/vehicle_0")
# Example: reset the agent at an arbitrary (x, y, yaw) position.
# yaw is in radians. This only affects the environment's internal
# bookkeeping by default; depending on your simulator integration this
# may or may not teleport the robot within the simulator engine.
start_x = 3.14
start_y = 2.71
start_yaw = math.pi / 2.0
obs, info = env.reset(position=(start_x, start_y, start_yaw))
start_x, start_y, start_yaw = start_x*2, start_y*2, start_yaw*2

for step in range(1000):
    action = env.action_space.sample()
    action = [1,1]  
 
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"---------- step {step} ----------")
    #print(f"obs = {obs}")
    print(f"action = {action}")
    print(f"reward = {reward}")
    print(f"terminated = {terminated}")
    print(f"truncated = {truncated}")
    print(f"info = {info}")
    sleep(0.1)
    if terminated or truncated:
        print("$$$$$$$$$$ ENV RESETTING $$$$$$$$$$$")
        obs, info = env.reset(position=(start_x, start_y, start_yaw))
        start_x, start_y, start_yaw = 0.314, 0.271, start_yaw+1

env.robot.camera.stop()
env.robot.motors.stop()