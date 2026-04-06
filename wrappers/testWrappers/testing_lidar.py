# -*- coding: utf-8 -*-
import time
import numpy as np

from wrappers.env_3d import DroneSwarm3D
from envs.createObjects import EnvObject
from config.defaults import (
    POS_SLICE,
    TARGET_SLICE,
    TARGET_REACH_DISTANCE,
    LIDAR_NUM_RAYS,
    LIDAR_NUM_LAYERS,
)


def unit_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


env = DroneSwarm3D(
    world_scale=5.0,
    num_drones_per_swarm=1,
    num_static_obstacles=0,
    num_dynamic_obstacles=0,
    max_steps=500,
    lidar=True,
    render_mode="human",
)

obs, info = env.reset(seed=42)

agent = "drone_0"

# --- get start + target ---
drone_pos0 = obs[agent][POS_SLICE].astype(np.float32)
to_target0 = obs[agent][TARGET_SLICE].astype(np.float32)
target_pos = drone_pos0 + to_target0

# --- place one obstacle between drone and target ---
path_dir = unit_vec(target_pos - drone_pos0)
path_dist = float(np.linalg.norm(target_pos - drone_pos0))

# Put obstacle roughly 40% of the way to target, but not too close
obs_dist = min(max(1.5, 0.4 * path_dist), max(1.5, path_dist - 1.0))
obstacle_pos = drone_pos0 + path_dir * obs_dist

# Slightly enlarge obstacle so rays hit it more easily
box = EnvObject(
    start_pos=obstacle_pos.tolist(),
    size=2.0,
    orientation=[0.0, 0.0, 0.0],
)
box.create("box", physics_call=env._call_physics, color=(1, 0, 0, 1))
env._obstacles.append(box)

print("Initial drone pos :", np.round(drone_pos0, 3))
print("Target pos        :", np.round(target_pos, 3))
print("Obstacle pos      :", np.round(obstacle_pos, 3))
print("Distance to target:", round(path_dist, 3))
print()

done = False
step_idx = 0

while not done:
    drone_pos = obs[agent][POS_SLICE].astype(np.float32)
    to_target = obs[agent][TARGET_SLICE].astype(np.float32)
    dist = float(np.linalg.norm(to_target))
    direction = unit_vec(to_target)

    # move straight toward target
    if dist < TARGET_REACH_DISTANCE:
        action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    else:
        action = np.array(
            [direction[0], direction[1], direction[2], 1.0], dtype=np.float32
        )

    obs, rew, term, trunc, info = env.step({agent: action})
    step_idx += 1

    # LiDAR part starts after first 18 base obs entries
    lidar = obs[agent][18:]
    lidar_grid = lidar.reshape(LIDAR_NUM_LAYERS, LIDAR_NUM_RAYS)

    # middle vertical layer
    mid_layer = lidar_grid[LIDAR_NUM_LAYERS // 2]

    # rays around the front-middle of the horizontal FOV
    center_idx = LIDAR_NUM_RAYS // 2
    front_window = mid_layer[
        max(0, center_idx - 2) : min(LIDAR_NUM_RAYS, center_idx + 3)
    ]

    print(
        f"step={step_idx:03d} | "
        f"dist={dist:.3f} | "
        f"reward={rew[agent]:.3f} | "
        f"term={term[agent]} | trunc={trunc[agent]} | "
        f"lidar_min={float(np.min(mid_layer)):.3f} | "
        f"front_window={np.round(front_window, 3)}"
    )

    done = bool(term[agent] or trunc[agent])
    time.sleep(1 / 30)

print("\nEpisode finished.")
print("Final info:", info[agent])

env.close()
