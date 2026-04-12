# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Optional
import numpy as np

from stable_baselines3.common.vec_env import VecEnvWrapper


class SB3VecResetTupleFix(VecEnvWrapper):
    """
    Fixes Gymnasium/SuperSuit incompatibilities with SB3:

    1) reset() returns (obs, info) -> SB3 expects obs only
    2) step_wait() sometimes returns (obs, rewards, terminations, truncations, infos)
       -> SB3 expects (obs, rewards, dones, infos)

    Stores reset-info in self.last_reset_info.
    """

    def __init__(self, venv):
        super().__init__(venv)
        self.last_reset_info: Optional[Any] = None

    def reset(self, **kwargs):
        out = self.venv.reset(**kwargs)

        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            self.last_reset_info = info
            return obs

        self.last_reset_info = None

        return out

    def step_async(self, actions) -> None:
        self.venv.step_async(actions)

    def step_wait(self):
        out = self.venv.step_wait()

        if isinstance(out, tuple) and len(out) == 5:
            obs, rewards, terminations, truncations, infos = out

            terminations = np.asarray(terminations, dtype=np.bool_)
            truncations = np.asarray(truncations, dtype=np.bool_)

            if terminations.ndim == 0:
                terminations = terminations.reshape(1)
            if truncations.ndim == 0:
                truncations = truncations.reshape(1)

            dones = np.logical_or(terminations, truncations).astype(
                np.bool_, copy=False
            )

            if infos is not None:
                for i in range(min(len(infos), len(dones))):
                    if isinstance(infos[i], dict):
                        infos[i]["terminated"] = bool(terminations[i])
                        infos[i]["truncated"] = bool(truncations[i])

            if dones.any():
                for i in range(len(dones)):
                    if dones[i] and isinstance(infos[i], dict):
                        infos[i]["terminal_observation"] = obs[i].copy()

                new_obs = self.venv.reset()
                if isinstance(new_obs, tuple):
                    new_obs, _ = new_obs

                for i in range(len(dones)):
                    if dones[i]:
                        obs[i] = new_obs[i]

            return obs, rewards, dones, infos

        if isinstance(out, tuple) and len(out) == 4:
            obs, rewards, dones, infos = out

            dones = np.asarray(dones, dtype=np.bool_)
            rewards = np.asarray(rewards, dtype=np.float32)

            if dones.ndim == 0:
                dones = dones.reshape(1)
            if rewards.ndim == 0:
                rewards = rewards.reshape(1)

            if infos is not None:
                for i in range(min(len(infos), len(dones))):
                    if isinstance(infos[i], dict):
                        infos[i]["terminated"] = bool(infos[i].get("terminated", False))

                        if "truncated" in infos[i]:
                            infos[i]["truncated"] = bool(infos[i]["truncated"])
                        elif "TimeLimit.truncated" in infos[i]:
                            infos[i]["truncated"] = bool(
                                infos[i]["TimeLimit.truncated"]
                            )
                        else:
                            infos[i]["truncated"] = False

            if dones.any():
                for i in range(len(dones)):
                    if dones[i] and isinstance(infos[i], dict):
                        infos[i]["terminal_observation"] = obs[i].copy()

                new_obs = self.venv.reset()
                if isinstance(new_obs, tuple):
                    new_obs, _ = new_obs

                for i in range(len(dones)):
                    if dones[i]:
                        obs[i] = new_obs[i]

            assert dones.dtype == np.bool_, f"dones dtype is {dones.dtype}"
            assert dones.shape == (
                self.num_envs,
            ), f"dones shape is {dones.shape}, expected {(self.num_envs,)}"

            return obs, rewards, dones, infos
