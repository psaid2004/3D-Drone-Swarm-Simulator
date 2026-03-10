from __future__ import annotations
from typing import Dict
import numpy as np

from config.defaults import (
    PROG_COEF,
    TIME_PEN,
    CNTRL_PEN,
    NEAR_OBS_PEN,
    GOAL_BONUS,
    COLLISION_PEN,
)

def compute_rewards(env, obs_dict, sim_actions: np.ndarray, step_state) -> Dict[str, float]:
    """Compute rewards given observations and actions."""

    reward_dict: Dict[str, float] = {}

    for i, agent in enumerate(env.possible_agents):
        if i >= env.total_num_drones:
            reward_dict[agent] = 0.0
            continue

        pos = obs_dict[agent][0:3]
        target = (
            env._targets[i].start_pos
            if env._targets is not None
            else np.zeros(3, dtype=np.float32)
        )

        dist = float(np.linalg.norm(target - pos))
        prev = float(env._prev_dists[i])
        env._prev_dists[i] = dist

        r_progress = PROG_COEF * (prev - dist)
        r_goal_dist = -0.15 * dist
        r_time = -TIME_PEN
        r_ctrl = -CNTRL_PEN * float(np.linalg.norm(sim_actions[i]))

        if len(env._obstacles) > 0:
            d_obs = float(np.linalg.norm(obs_dict[agent][9:12]))
            r_near = -NEAR_OBS_PEN / max(d_obs, 0.10)
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

        if step_state.just_reached[i]:
            r += GOAL_BONUS

        if step_state.collisions[i]:
            r -= COLLISION_PEN
        elif np.any(step_state.collisions):
            r -= 0.2 * COLLISION_PEN

        reward_dict[agent] = float(r)

    return reward_dict
