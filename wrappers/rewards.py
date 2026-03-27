from __future__ import annotations
from typing import Dict
import numpy as np

from config.defaults import (
    PROG_COEF,
    TIME_PEN,
    CNTRL_PEN,
    NEAR_OBS_PEN,
    NEAR_FRONT_OBS_PEN,
    NEAR_SIDE_OBS_PEN,
    GOAL_BONUS,
    DRONE_COLLISION_PEN,
    OBSTACLE_COLLISION_PEN,
    WALL_COLLISION_PEN,
    POS_SLICE,
    OBS_SLICE,
    DIST_PEN,
    TRUNC_PEN,
    ZERO_SPEED_PEN,
    LIDAR_NUM_LAYERS,
    LIDAR_NUM_RAYS,
    FINAL_MILE_COEF,
    FINAL_MILE_R,
    ALIGN_COEF,
)

def compute_rewards(env, obs_dict, sim_actions: np.ndarray, step_state) -> Dict[str, float]:
    """Compute rewards given observations and actions."""

    reward_dict: Dict[str, float] = {}
    reward_info: Dict[str, dict] = {}

    for i, agent in enumerate(env.possible_agents):
        if i >= env.total_num_drones:
            reward_dict[agent] = 0.0
            continue

        # --------------- 1. Terminal one-time outcomes ---------------

        # one-time collision penalty
        if step_state.wall_collisions[i]:
            reward_dict[agent] = float(-WALL_COLLISION_PEN)
            continue
        if step_state.drone_collisions[i]:
            reward_dict[agent] = float(-DRONE_COLLISION_PEN)
            continue
        if step_state.obstacle_collisions[i]:
            reward_dict[agent] = float(-OBSTACLE_COLLISION_PEN)
            continue

        # one-time goal reached reward
        if step_state.reached_now[i]:
            reward_dict[agent] = float(GOAL_BONUS)
            continue

        # no more updates to rewards for inactive drones
        if env._dead[i] or env._reached_once[i] or env._despawned[i]:
            reward_dict[agent] = 0.0
            continue

        # --------------- 2. Goal geometry ---------------

        pos = obs_dict[agent][POS_SLICE]
        target = (
            env._targets[i].start_pos
            if env._targets is not None
            else np.zeros(3, dtype=np.float32)
        )

        to_target_vec = target - pos
        dist_to_target = float(np.linalg.norm(to_target_vec))

        prev_dist_to_target = float(env._prev_dists[i])
        env._prev_dists[i] = dist_to_target

        to_target_norm = np.linalg.norm(to_target_vec)

        if to_target_norm > 1e-6:
            target_dir = to_target_vec / to_target_norm
            move_dir = sim_actions[i][0:3]
            # values between -1 and 1
            alignment = float(np.dot(move_dir, target_dir))
        else:
            alignment = 0.0


        # --------------- 3. Progress, distance, time, and action cost ---------------

        if env._step_count <= 1:
            r_progress = 0.0
        else:
            r_progress = PROG_COEF * (prev_dist_to_target - dist_to_target) # reward for change in distance to target

        if not env.lidar:
            # easier reward during early curriculum
            r_distance_pen = -0.25 * DIST_PEN * dist_to_target
            r_time = -0.25 * TIME_PEN
            r_ctrl = 0.0
        else:
            r_distance_pen = -DIST_PEN * dist_to_target
            r_time = -TIME_PEN
            r_ctrl = -CNTRL_PEN * float(np.linalg.norm(sim_actions[i]))


        # --------------- 4. Alignment reward ---------------

        vel = env._env._getDroneStateVector(i)[10:13]
        speed = float(np.linalg.norm(vel))

        # +0.5 if moving directly toward target, else - 0.5
        r_align = 0.0 if not env.lidar else ALIGN_COEF * alignment

        # --------------- 5. Final-mile shaping ---------------

        # Add a small bonus when the drone is within some radius of the goal, increasing smoothly as it gets closer.
        if env.lidar and dist_to_target < FINAL_MILE_R:
            r_final = FINAL_MILE_COEF * (1.0 - (dist_to_target / FINAL_MILE_R) ** 2)
        else:
            r_final = 0.0

        # --------------- 6. Tangential velocity penalty near goal ---------------

        vel = np.array(env._env._getDroneStateVector(i)[10:13], dtype=np.float32)
        speed = float(np.linalg.norm(vel))

        if to_target_norm > 1e-6 and speed > 1e-6:
            target_dir = to_target_vec / to_target_norm
            vel_parallel = np.dot(vel, target_dir) * target_dir
            vel_tangent = vel - vel_parallel
            tangential_speed = float(np.linalg.norm(vel_tangent))
        else:
            tangential_speed = 0.0

        if env.lidar and dist_to_target < FINAL_MILE_R:
            r_tan = -0.15 * tangential_speed
        else:
            r_tan = 0.0

        # --------------- 7. Obstacle avoidance shaping ---------------

        r_near = 0.0

        if env.lidar:
            scan = obs_dict[agent][env._base_obs_dim:]
            scan_2d = scan.reshape(LIDAR_NUM_LAYERS, LIDAR_NUM_RAYS)

            front_idx = np.array([23, 0, 1], dtype=np.int32)
            left_idx = np.array([4, 5, 6, 7, 8], dtype=np.int32)
            right_idx = np.array([16, 17, 18, 19, 20], dtype=np.int32)

            # take closest return across all vertical layers
            front_min = float(np.min(scan_2d[:, front_idx]))
            left_min = float(np.min(scan_2d[:, left_idx]))
            right_min = float(np.min(scan_2d[:, right_idx]))

            if front_min < 0.35:
                r_near = -NEAR_FRONT_OBS_PEN * (0.35 - front_min) / 0.35
            elif min(left_min, right_min) < 0.25:
                d_obs = min(left_min, right_min)
                r_near = -NEAR_SIDE_OBS_PEN * (0.25 - d_obs) / 0.25
            else:
                r_near = 0.0

        elif len(env._obstacles) > 0:
            d_obs = float(np.linalg.norm(obs_dict[agent][OBS_SLICE]))
            r_near = -NEAR_OBS_PEN / max(d_obs, 0.10)

        else:
            r_near = 0.0

        # --------------- 8. Combine shaping terms ---------------

        r = (
            r_progress
            + r_distance_pen
            + r_time
            + r_ctrl
            + r_near
            + r_align
            + r_tan
            + r_final
            + step_state.cohesion_pen
            + step_state.sep_pen
        )
        r_info = {"r_progress": r_progress}

        # --------------- 9. Extra penalties ---------------

        # penalty for if time limit was reached
        if step_state.time_limit:
            if env.lidar:
                r -= TRUNC_PEN
            else:
                r -= 0.25 * TRUNC_PEN

        # penalty for if drone has no / little speed (is not moving)
        if speed < 0.1:
            r -= ZERO_SPEED_PEN



        # print(r_info)
        # print(r)


        reward_dict[agent] = float(r)
        reward_info[agent] = r_info

    return reward_dict
