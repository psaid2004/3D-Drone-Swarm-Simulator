from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from davids_drones.wrappers.step_types import StepState
from davids_drones.config.defaults import (
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
            continue

        if (env._dead[i] or env._reached_once[i] or env._despawned[i]) and a in actions:
            actions[a] = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def convert_actions_for_simulator(env, actions: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert normalised policy actions into simulator (VelocityAviary) velocity commands.

    Policy output = [dir_x, dir_y, dir_z, speed] in [-1, 1]
    Simulator expects: direction (unit vector) and speed (m/s)
    """
    act = np.zeros((env.total_num_drones, 4), dtype=np.float32)

    for i, a in enumerate(env.possible_agents):

        # inactive drones -> just hover
        if i >= env.total_num_drones:
            act[i] = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            continue

        # default is direction = up and speed = 0 m/s
        dir_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        speed = 0.0

        if a in actions:
            u = np.array(actions[a], dtype=np.float32).reshape(-1)
            # clip to valid range
            u = np.clip(u[:4], -1.0, 1.0)

            # get the required direction based on normalised action
            v = u[:3].astype(np.float32)
            n = np.linalg.norm(v)
            dir_vec = v / n if n > 1e-6 else dir_vec

            # speed mapping: u[3] from [-1,1] -> [0,1 (vel_scale)]
            speed01 = float((u[3] + 1.0) * 0.5)
            speed = env._env.SPEED_LIMIT * speed01  # m/s

        # retrieve direction vector and speed for drone
        act[i, 0:3] = dir_vec
        act[i, 3] = float(speed)

    return act


def compute_step_state(env, obs_dict, actions: Dict[str, np.ndarray]) -> StepState:
    """ Retrieve status of step (only considers active drones). """

    time_limit = bool(env._step_count >= env.max_steps)

    collisions, wall_collisions, obstacle_collisions, drone_collisions = env.check_collisions_per_drone()
    if isinstance(collisions, bool):
        collisions = np.zeros(env.total_num_drones, dtype=np.bool_)
        wall_collisions = np.zeros(env.total_num_drones, dtype=np.bool_)
        obstacle_collisions = np.zeros(env.total_num_drones, dtype=np.bool_)
        drone_collisions = np.zeros(env.total_num_drones, dtype=np.bool_)

    env._wall_collision_once = np.logical_or(env._wall_collision_once, wall_collisions)
    env._obstacle_collision_once = np.logical_or(env._obstacle_collision_once, obstacle_collisions)
    env._drone_collision_once = np.logical_or(env._drone_collision_once, drone_collisions)

    # mark drones that are dead
    env._dead = np.logical_or(env._dead, collisions)

    # despawn newly collided drones
    env.despawn_newly_collided_drones(collisions)

    # detect newly-reached goals for one-time bonuses in reward
    was_reached = env._reached_once.copy()
    reached_now = np.zeros((env.total_num_drones,), dtype=np.bool_)

    for i, a in enumerate(env.possible_agents):
        if i >= env.total_num_drones:
            continue

        # skip drones if they are already dead/despawned before goal check
        if env._dead[i] and env._despawned[i]:
            continue

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

    # determine which drones just reached the target in this step
    just_reached = np.logical_and(reached_now, np.logical_not(was_reached))

    # obtain active drone positions for ongoing swarm logic
    active_idx = env.active_drone_indices()
    positions = env.active_drone_positions()

    all_collided = bool(np.all(env._dead[:env.total_num_drones]))

    max_sep, min_sep, too_far, too_close = env._pairwise_sep_stats(positions)

    # Swarm shaping penalties (penalty for when drones in swarm spread out too far or get too close)
    cohesion_pen = 0.0
    if positions.shape[0] > 1 and max_sep > DEFAULT_SWARM_MAX_SEP:
        cohesion_pen = - COHESION_PEN * (max_sep - DEFAULT_SWARM_MAX_SEP)
    sep_pen = 0.0
    if positions.shape[0] > 1 and min_sep < DEFAULT_SWARM_MIN_SEP:
        sep_pen = -SEPARATION_PEN * (DEFAULT_SWARM_MIN_SEP - min_sep)

    no_active_drones_left = len(active_idx) == 0

    # episode ends if all collided or no more active drones present (episode is truncated below if time limit is reached)
    episode_done = bool(all_collided or no_active_drones_left)

    return StepState(
        time_limit=time_limit,
        collisions=collisions,
        wall_collisions=wall_collisions,
        obstacle_collisions=obstacle_collisions,
        drone_collisions=drone_collisions,
        all_collided=all_collided,
        positions=positions,
        max_sep=float(max_sep),
        min_sep=float(min_sep),
        too_far=bool(too_far),
        too_close=bool(too_close),
        reached_now=reached_now,
        just_reached=just_reached,
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
            term_dict[agent] = bool(step_state.episode_done)
            trunc_dict[agent] = bool(step_state.time_limit)
            continue

        term_dict[agent] = bool(
            env._dead[i] or env._reached_once[i] or env._despawned[i] or step_state.episode_done
        )
        trunc_dict[agent] = bool(step_state.time_limit)

    return term_dict, trunc_dict


def build_info_dict(env, step_state) -> dict:
    """ Build info dictionary. """
    info_dict = {}

    for i, agent in enumerate(env.possible_agents):
        if i >= env.total_num_drones:
            info_dict[agent] = {
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
                "episode_done_all_reached": bool(step_state.no_active_drones_left),
            }
            continue

        info_dict[agent] = {
            "step": int(env._step_count),
            "active": True,
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
            "just_reached": bool(step_state.just_reached[i]),
            "swarm_max_sep": float(step_state.max_sep),
            "swarm_min_sep": float(step_state.min_sep),
            "swarm_too_far": bool(step_state.too_far),
            "swarm_too_close": bool(step_state.too_close),
            "episode_done_collision": bool(step_state.collisions[i]),
            "episode_done_all_reached": bool(step_state.no_active_drones_left),
        }

    return info_dict





