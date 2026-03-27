from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from wrappers.step_types import StepState
from config.defaults import (
    DEFAULT_SWARM_MAX_SEP,
    DEFAULT_SWARM_MIN_SEP,
    COHESION_PEN,
    SEPARATION_PEN,
    GOAL_REACH_DISTANCE,
    POS_SLICE
)


def freeze_inactive_actions(env, actions: Dict[str, np.ndarray]) -> None:
    """Freeze actions for drones that are no longer active in the episode."""

    # - IMPORTANT: to make speed=0 in the mapping, we must set u[3] = -1
    #   because u[3] in [-1,1] maps to speed01 in [0,1] by:
    #     speed01 = (u[3] + 1) * 0.5

    for i, a in enumerate(env.possible_agents):
        if i >= env.total_num_drones:
            actions[a] = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
            continue

        if (env._dead[i] or env._reached_once[i] or env._despawned[i]) and a in actions:
            actions[a] = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def convert_actions_for_simulator(env, actions: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert policy actions to the normalized format expected by VelocityAviary.

    Output format per drone:
        [dir_x, dir_y, dir_z, speed_scale]

    where:
        dir_* is a unit direction vector
        speed_scale is in [0, 1]
    """
    act = np.zeros((env.max_total_num_drones, 4), dtype=np.float32)

    for i, a in enumerate(env.possible_agents):
        if i >= env.total_num_drones:
            continue

        # default: no movement
        dir_vec = np.zeros(3, dtype=np.float32)
        speed_scale = 0.0

        if a in actions:
            u = np.array(actions[a], dtype=np.float32).reshape(-1)
            u = np.clip(u[:4], -1.0, 1.0)

            v = u[:3].astype(np.float32)
            dir_vec = np.clip(v, -1.0, 1.0)
            speed_scale = float((u[3] + 1.0) * 0.5)  # map [-1,1] -> [0,1]

        act[i, 0:3] = dir_vec
        act[i, 3] = speed_scale

    return act


def compute_step_state(env, obs_dict) -> StepState:
    """ Retrieve status of step (only considers active drones). """

    time_limit = bool(env._step_count >= env.max_steps)

    collisions, wall_collisions, obstacle_collisions, drone_collisions = env.check_collisions_per_drone()

    if isinstance(collisions, bool):
        collisions = np.zeros(env.total_num_drones, dtype=np.bool_)
        wall_collisions = np.zeros(env.total_num_drones, dtype=np.bool_)
        obstacle_collisions = np.zeros(env.total_num_drones, dtype=np.bool_)
        drone_collisions = np.zeros(env.total_num_drones, dtype=np.bool_)

    # Expand active-only arrays to fixed-capacity arrays
    collisions_full = np.zeros(env.max_total_num_drones, dtype=np.bool_)
    wall_collisions_full = np.zeros(env.max_total_num_drones, dtype=np.bool_)
    obstacle_collisions_full = np.zeros(env.max_total_num_drones, dtype=np.bool_)
    drone_collisions_full = np.zeros(env.max_total_num_drones, dtype=np.bool_)

    collisions_full[:env.total_num_drones] = collisions
    wall_collisions_full[:env.total_num_drones] = wall_collisions
    obstacle_collisions_full[:env.total_num_drones] = obstacle_collisions
    drone_collisions_full[:env.total_num_drones] = drone_collisions

    env._wall_collision_once = np.logical_or(env._wall_collision_once, wall_collisions_full)
    env._obstacle_collision_once = np.logical_or(env._obstacle_collision_once, obstacle_collisions_full)
    env._drone_collision_once = np.logical_or(env._drone_collision_once, drone_collisions_full)

    # mark drones that are dead
    env._dead = np.logical_or(env._dead, collisions_full)

    # despawn newly collided drones
    env.despawn_newly_collided_drones(collisions_full)

    # detect newly-reached goals for one-time bonuses in reward
    reached_now = np.zeros((env.max_total_num_drones,), dtype=np.bool_)

    for i, a in enumerate(env.possible_agents):
        if i >= env.total_num_drones:
            continue

        # skip drones if they are already dead/despawned before goal check
        if env._dead[i] or env._despawned[i]:
            continue # TODO THIS WAS AND PREVIOUSLY BUT I THINK IT SHOULD BE OR?

        pos = obs_dict[a][POS_SLICE]
        target = (
            env._targets[i].start_pos
            if env._targets is not None
            else np.zeros(3, dtype=np.float32)
        )
        dist = float(np.linalg.norm(target - pos))

        if dist < GOAL_REACH_DISTANCE:
            reached_now[i] = True
            env._reached_once[i] = True

    # obtain active drone positions for ongoing swarm logic
    active_idx = env.active_drone_indices()
    positions = env.active_drone_positions()

    all_collided = bool(np.all(env._dead[:env.total_num_drones]))
    all_reached = bool(np.all(env._reached_once[:env.total_num_drones]))

    max_sep, min_sep, too_far, too_close = env._pairwise_sep_stats(positions)

    # Swarm shaping penalties (penalty for when drones in swarm spread out too far or get too close)
    cohesion_pen = 0.0
    if positions.shape[0] > 1 and max_sep > DEFAULT_SWARM_MAX_SEP:
        cohesion_pen = - COHESION_PEN * (max_sep - DEFAULT_SWARM_MAX_SEP)
    sep_pen = 0.0
    if positions.shape[0] > 1 and min_sep < DEFAULT_SWARM_MIN_SEP:
        sep_pen = -SEPARATION_PEN * (DEFAULT_SWARM_MIN_SEP - min_sep)

    no_active_drones_left = len(active_idx) == 0

    # episode ends if all collided, all reached, or there are no active drones left
    episode_done = bool(all_collided or all_reached or no_active_drones_left)

    return StepState(
        time_limit=time_limit,
        collisions=collisions_full,
        wall_collisions=wall_collisions_full,
        obstacle_collisions=obstacle_collisions_full,
        drone_collisions=drone_collisions_full,
        all_collided=all_collided,
        all_reached=all_reached,
        positions=positions,
        max_sep=float(max_sep),
        min_sep=float(min_sep),
        too_far=bool(too_far),
        too_close=bool(too_close),
        reached_now=reached_now,
        no_active_drones_left=bool(no_active_drones_left),
        episode_done=episode_done,
        cohesion_pen=float(cohesion_pen),
        sep_pen=float(sep_pen),
    )


def compute_terminations(env, step_state) -> Tuple[dict, dict]:
    """ Compute terminations. """
    term_dict = {} # whether agent terminated
    trunc_dict = {} # whether agent is truncated by time limit

    for i, agent in enumerate(env.possible_agents):
        if i >= env.total_num_drones:
            term_dict[agent] = False
            trunc_dict[agent] = False
            continue

        term_dict[agent] = bool(
            env._dead[i] or env._reached_once[i] or step_state.episode_done
        )
        trunc_dict[agent] = bool(step_state.time_limit)

    return term_dict, trunc_dict


def build_info_dict(env, step_state) -> dict:
    """ Build info dictionary. """
    info_dict = {}

    for i, agent in enumerate(env.possible_agents):
        if i >= env.total_num_drones:
            info_dict[agent] = {
                "drone_idx": int(i),
                "step": int(env._step_count),
                "active": False,
                "collided": False,
                "wall_collision": False,
                "obstacle_collision": False,
                "drone_collision": False,
                "wall_collision_now": False,
                "obstacle_collision_now": False,
                "drone_collision_now": False,
                "reached": False,
                "swarm_max_sep": float(step_state.max_sep),
                "swarm_min_sep": float(step_state.min_sep),
                "swarm_too_far": bool(step_state.too_far),
                "swarm_too_close": bool(step_state.too_close),
                "episode_done_collision": False,
                "episode_done_all_reached": bool(step_state.all_reached),
            }
            continue

        pos = env._env._getDroneStateVector(i)[POS_SLICE]
        target = env._targets[i].start_pos if env._targets is not None else np.zeros(3, dtype=np.float32)
        dist_to_target = float(np.linalg.norm(target - pos))

        vel = np.array(env._env._getDroneStateVector(i)[10:13], dtype=np.float32)
        actual_speed = float(np.linalg.norm(vel))

        cmd_speed = float(env._last_sim_actions[i][3]) if hasattr(env, "_last_sim_actions") else 0.0

        is_active = bool(not (env._dead[i] or env._reached_once[i] or env._despawned[i]))

        info_dict[agent] = {
            "drone_idx": int(i),
            "step": int(env._step_count),
            "active": is_active,
            "collided": bool(env._dead[i]),
            "wall_collision": bool(env._wall_collision_once[i]),
            "obstacle_collision": bool(env._obstacle_collision_once[i]),
            "drone_collision": bool(env._drone_collision_once[i]),
            "collision_now": bool(step_state.collisions[i]),
            "wall_collision_now": bool(step_state.wall_collisions[i]),
            "obstacle_collision_now": bool(step_state.obstacle_collisions[i]),
            "drone_collision_now": bool(step_state.drone_collisions[i]),
            "reached": bool(env._reached_once[i]),
            "reached_now": bool(step_state.reached_now[i]),
            "dist_to_target": dist_to_target,
            "cmd_speed": cmd_speed,
            "actual_speed": actual_speed,
            "swarm_max_sep": float(step_state.max_sep),
            "swarm_min_sep": float(step_state.min_sep),
            "swarm_too_far": bool(step_state.too_far),
            "swarm_too_close": bool(step_state.too_close),
            "episode_done_collision": bool(step_state.collisions[i]),
            "episode_done_all_reached": bool(step_state.all_reached),
        }

    return info_dict





