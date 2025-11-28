from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv
from time import sleep
import math

x_min = 0
x_max = 1.6
y_min = 0
y_max = 1.6

def is_out_of_bounds(pose):
    if pose is None:
        return False
    x = pose["position"]["x"]
    y = pose["position"]["y"]
    return x < x_min or x > x_max or y < y_min or y > y_max


def pose_matches_target(pose, target, pos_tol=0.1, yaw_tol=0.1):
    if pose is None or target is None:
        return False
    tx, ty, tyaw = target
    px = pose["position"]["x"]
    py = pose["position"]["y"]
    # yaw from quaternion (z-up)
    w = pose["rotation"]["w"]
    x = pose["rotation"]["x"]
    y = pose["rotation"]["y"]
    z = pose["rotation"]["z"]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    yaw_err = math.atan2(math.sin(yaw - tyaw), math.cos(yaw - tyaw))
    return abs(px - tx) <= pos_tol and abs(py - ty) <= pos_tol #and abs(yaw_err) <= yaw_tol

env = DuckiematrixDB21JEnv(entity_name="map_0/vehicle_0")

start_x = 0.314
start_y = 0.271
start_yaw = -0.
desired_reset = (start_x, start_y, start_yaw)
obs, info = env.reset(position=desired_reset)
start_x, start_y, start_yaw = start_x*2, start_y*2, start_yaw*2

for step in range(1000):

    action = [1, 1]
    obs, reward, terminated, truncated, info = env.step(action)
    print(time)
    action = model(obs)
    print()
    print(f"---------- step {step} ----------")
    print(f"action = {action}")
    print(f"reward = {reward}")
    print(f"terminated = {terminated}")
    print(f"truncated = {truncated}")
    print(f"info = {info}")
    sleep(0.1)
    if info is not None:
        if terminated or truncated or is_out_of_bounds(info.get("pose")):
            print("$$$$$$$$$$ ENV RESETTING $$$$$$$ :^) $$$$")
            i=0
            while not pose_matches_target(info.get("pose"), desired_reset):
                print("----- reset attempt ", i, " -----")
                print(pose_matches_target(info.get("pose"), desired_reset))
                print("info.get(pose) = ", info.get("pose")["position"]["x"], info.get("pose")["position"]["y"])
                print("desired_reset = ", desired_reset)
                print()
                obs, info = env.reset(position=desired_reset)
                sleep(0.1)
                i+=1
            desired_reset = 0.314, 0.271, start_yaw + 0.5
            start_yaw += 0.5

env.robot.camera.stop()
env.robot.motors.stop()
