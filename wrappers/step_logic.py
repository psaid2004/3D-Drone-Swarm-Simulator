from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from wrappers.step_types import StepState
from config.defaults import (
    DEFAULT_TERMINATE_ON_COHESION_BREAK,
    DEFAULT_TERMINATE_ON_SEPARATION,
    DEFAULT_SWARM_MAX_SEP,
    DEFAULT_SWARM_MIN_SEP,
    COHESION_PEN,
    SEPARATION_PEN,
    GOAL_REACH_DISTANCE,
)


def freeze_reached_actions(env, actions: Dict[str, np.ndarray]) -> None:
    """ Freeze/ despawn drones that have reached the target. """ # TODO MAKE LOGIC CORRECT

    # - IMPORTANT: to make speed=0 in the mapping, we must set u[3] = -1
    #   because u[3] in [-1,1] maps to speed01 in [0,1] by:
    #     speed01 = (u[3] + 1) * 0.5
    for i, a in enumerate(env.possible_agents):
        if i < env.total_num_drones and env._reached_once[i] and a in actions:
            actions[a] = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def convert_actions_for_simulator(env, actions: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert normalised policy actions into simulator (VelocityAviary) velocity commands.

    Policy output = [dir_x, dir_y, dir_z, speed] in [-1, 1]
    Simulator expects: direction (unit vector) and speed (m/s)
    """
    from config.defaults import TAKEOFF_STEPS, TAKEOFF_VZ, HOVER_BIAS_VZ

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

            # hover bias to fight gravity drift
            dir_vec[2] += HOVER_BIAS_VZ
            dir_vec /= max(np.linalg.norm(dir_vec), 1e-6)

        # automatic takeoff phase (override RL for initial steps during training and tell drone to go up with takeoff velocity (off ground))
        if env._step_count <= TAKEOFF_STEPS:
            dir_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            speed = float(abs(TAKEOFF_VZ))

        # retrieve direction vector and speed for drone
        act[i, 0:3] = dir_vec
        act[i, 3] = float(speed)

    return act


def compute_step_state(env, obs_dict, actions: Dict[str, np.ndarray]) -> StepState:
    """ Retrieve status of step. """

    time_limit = bool(env._step_count >= env.max_steps)

    collisions = env.check_collisions_per_drone()
    if isinstance(collisions, bool):
        collisions = np.zeros(env.total_num_drones, dtype=np.bool_)

    all_collided = bool(np.all(collisions[:env.total_num_drones]))

    positions = env._active_positions()
    max_sep, min_sep, too_far, too_close = env._pairwise_sep_stats(positions)

    swarm_term = (too_far and DEFAULT_TERMINATE_ON_COHESION_BREAK) or (
            too_close and DEFAULT_TERMINATE_ON_SEPARATION)

    # Swarm shaping penalties
    cohesion_pen = 0.0
    if positions.shape[0] > 1 and max_sep > DEFAULT_SWARM_MAX_SEP:
        cohesion_pen = - COHESION_PEN * (max_sep - DEFAULT_SWARM_MAX_SEP)

    sep_pen = 0.0
    if positions.shape[0] > 1 and min_sep < DEFAULT_SWARM_MIN_SEP:
        sep_pen = -SEPARATION_PEN * (DEFAULT_SWARM_MIN_SEP - min_sep)

    # detect newly-reached goals for one-time bonuses in reward
    was_reached = env._reached_once.copy()
    reached_now = np.zeros((env.total_num_drones,), dtype=np.bool_)

    for i, a in enumerate(env.possible_agents):
        if i >= env.total_num_drones:
            continue

        pos = obs_dict[a][0:3]
        target = env._targets[i].start_pos if env._targets is not None else np.zeros(3,
                                                                                       dtype=np.float32)
        dist = float(np.linalg.norm(target - pos))

        if not env.dead[i] and dist < GOAL_REACH_DISTANCE:
            reached_now[i] = True
            env._reached_once[i] = True  # TODO I FEEL LIKE IF IT DIED SHOULD BE UPDATED HERE?

        # allow a crashed drone to stop moving
        if env.dead[i] and a in actions:
            actions[a] = np.array([0.0, 0.0, 0.0, -1.0],
                                  dtype=np.float32)

    just_reached = np.logical_and(reached_now, np.logical_not(was_reached))
    all_active_reached = all(bool(env._reached_once[i]) for i in range(env.total_num_drones))
    episode_done = bool(
        all_collided or all_active_reached or swarm_term)  # episode ends if everyone crashed, all active reached or spread too far

    env.dead = np.logical_or(env.dead, collisions)

    return StepState(
        time_limit=time_limit,
        collisions=collisions,
        all_collided=all_collided,
        positions=positions,
        max_sep=float(max_sep),
        min_sep=float(min_sep),
        too_far=bool(too_far),
        too_close=bool(too_close),
        swarm_term=swarm_term,
        reached_now=reached_now,
        just_reached=just_reached,
        all_active_reached=bool(all_active_reached),
        episode_done=episode_done,
        cohesion_pen=float(cohesion_pen),
        sep_pen=float(sep_pen),
    )


def compute_terminations(env, step_state) -> Tuple[dict, dict]:
    """ Compute terminations. """
    term_dict = {}
    trunc_dict = {}

    for i, agent in enumerate(env.possible_agents):
        if i >= env.total_num_drones:
            term_dict[agent] = bool(step_state.episode_done)
            trunc_dict[agent] = bool(step_state.time_limit)
            continue

        # allow crashed drone to terminate or if they have reached the goal
        term_dict[agent] = bool(
            env.dead[i] or step_state.episode_done or env._reached_once[i]
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
                "reached": False,
                "swarm_max_sep": float(step_state.max_sep),
                "swarm_min_sep": float(step_state.min_sep),
                "swarm_too_far": bool(step_state.too_far),
                "swarm_too_close": bool(step_state.too_close),
                "episode_done_collision": False,
                "episode_done_all_reached": bool(step_state.all_active_reached),
                "episode_done_swarm": bool(step_state.swarm_term),
            }
            continue

        info_dict[agent] = {
            "step": int(env._step_count),
            "active": True,
            "collided": bool(env.dead[i]),  # ever collided
            "collision_now": bool(step_state.collisions[i]),  # collided this step
            "reached": bool(env._reached_once[i]),
            "reached_now": bool(step_state.reached_now[i]),
            "just_reached": bool(step_state.just_reached[i]),
            "swarm_max_sep": float(step_state.max_sep),
            "swarm_min_sep": float(step_state.min_sep),
            "swarm_too_far": bool(step_state.too_far),
            "swarm_too_close": bool(step_state.too_close),
            "episode_done_collision": bool(step_state.collisions[i]),
            "episode_done_all_reached": bool(step_state.all_active_reached),
            "episode_done_swarm": bool(step_state.swarm_term),
        }

    return info_dict





