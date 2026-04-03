# -*- coding: utf-8 -*-
from stable_baselines3.common.vec_env import VecEnvWrapper
import numpy as np


class HandleInactiveDrones(VecEnvWrapper):
    """
    After VecNormalize in training, force inactive drone slots in vec to zero observations (consistent with env).
    This makes inactive slots look truly inert to policy.
    """

    def __init__(self, venv, active_drones: int, max_drones: int) -> None:
        super().__init__(venv)
        self.active_drones = int(active_drones)
        self.max_drones = int(max_drones)

        # total_slots = max_drones * number_of_envs
        self._inactive_mask = np.array(
            [
                (slot_idx % self.max_drones) >= self.active_drones
                for slot_idx in range(self.num_envs)
            ],
            dtype=np.bool_,
        )

    def _zero_inactive_rows(self, obs: np.ndarray) -> np.ndarray:
        """Add the active / inactive flag mask to the observation."""
        obs = np.array(obs, copy=True)
        obs[self._inactive_mask] = 0.0
        return obs

    def reset(self):
        obs = self.venv.reset()
        return self._zero_inactive_rows(obs)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        obs = self._zero_inactive_rows(obs)

        # keep info consistent too
        for i, is_inactive in enumerate(self._inactive_mask):
            if is_inactive and i < len(infos):
                infos[i] = dict(infos[i])
                infos[i]["active"] = False

        return obs, rewards, dones, infos
