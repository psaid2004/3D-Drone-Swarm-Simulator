from __future__ import annotations
from typing import Dict
import numpy as np

from config.defaults import (
    PROG_COEF,
    TIME_PEN,
    CNTRL_PEN,
    NEAR_OBS_PEN,
    GOAL_BONUS,
    DRONE_COLLISION_PEN,
    OBSTACLE_COLLISION_PEN,
    WALL_COLLISION_PEN,
    SHARED_COLLISION_PEN,
    POS_SLICE,
    OBS_SLICE,
)

def compute_rewards(env, obs_dict, sim_actions: np.ndarray, step_state) -> Dict[str, float]:
    """Compute rewards given observations and actions."""

    reward_dict: Dict[str, float] = {}

    for i, agent in enumerate(env.possible_agents):
        if i >= env.total_num_drones:
            reward_dict[agent] = 0.0
            continue

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
        if step_state.just_reached[i]:
            reward_dict[agent] = float(GOAL_BONUS)
            continue

        # if drone already inactive, no more updates to rewards
        if env._dead[i] or env._reached_once[i] or env._despawned[i]:
            reward_dict[agent] = 0.0
            continue

        pos = obs_dict[agent][POS_SLICE]
        target = (
            env._targets[i].start_pos
            if env._targets is not None
            else np.zeros(3, dtype=np.float32)
        )
        dist = float(np.linalg.norm(target - pos))
        prev = float(env._prev_dists[i])
        env._prev_dists[i] = dist

        r_progress = PROG_COEF * (prev - dist)
        r_goal_dist = -0.15 * dist # TODO WHAT IS THIS?
        r_time = -TIME_PEN
        r_ctrl = -CNTRL_PEN * float(np.linalg.norm(sim_actions[i])) # TODO WHAT IS THIS?

        if len(env._obstacles) > 0:
            d_obs = float(np.linalg.norm(obs_dict[agent][OBS_SLICE]))
            r_near = -NEAR_OBS_PEN / max(d_obs, 0.10) #  TODO WHY LIKE THIS?
        else:
            r_near = 0.0

        r = (
            r_progress
            + r_goal_dist
            + r_time
            + r_ctrl
            + r_near
            + step_state.cohesion_pen
            + step_state.sep_pen
        )

        # small team penalty if any collisions occur
        if np.any(step_state.collisions):
            r -= 0.2 * SHARED_COLLISION_PEN

        reward_dict[agent] = float(r)

    return reward_dict
