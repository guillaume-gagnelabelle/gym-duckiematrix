import numpy as np
import math

x_min = 0
x_max = 3 * 0.585
y_min = 0
y_max = 3 * 0.585


# vestigial from earlier experiments (reset function used to be more complex)
def pose_matches_target(pose, target, pos_tol=0.1, yaw_tol=0.1):
    if pose is None or target is None:
        return False
    
    tx, ty, tyaw = target
    px = pose["position"]["x"]
    py = pose["position"]["y"]

    yaw = compute_yaw(pose)
    yaw_err = math.atan2(math.sin(yaw - tyaw), math.cos(yaw - tyaw))
    return abs(px - tx) <= pos_tol and abs(py - ty) <= pos_tol #and abs(yaw_err) <= yaw_tol

def print_step_info(step, action, reward, terminated, d, theta, info):
    print()
    print(f"------------------------------ step {step} (before rollout) ------------------------------")
    print(f"action = {action}")
    print(f"reward = {reward:.2f}")
    print(f"terminated = {terminated}")
    print(f"d = {d:.2f}")
    print(f"theta = {theta:.2f}")
    if info is not None:
        print(f"info = ({round(info['pose']['position']['x'], 2)}, {round(info['pose']['position']['y'], 2)}, {round(compute_yaw(info['pose']), 2)})")

def is_out_of_bounds(pose):
    if pose is None:
        return False
    x = pose["position"]["x"]
    y = pose["position"]["y"]
    return x < x_min or x > x_max or y < y_min or y > y_max


def quaternion_to_euler(q):
    w, x, y, z = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.pi / 2 * np.sign(sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw  # in radians

def compute_yaw(pose):
    if pose is None:
        return None
    w = pose["rotation"]["w"]
    x = pose["rotation"]["x"]
    y = pose["rotation"]["y"]
    z = pose["rotation"]["z"]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return yaw