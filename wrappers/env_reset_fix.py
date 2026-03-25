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

    Stores reset-info in self.last_reset_info (optional).
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
            dones = np.logical_or(terminations, truncations)

            if infos is not None:
                for i in range(len(infos)):
                    if isinstance(infos[i], dict):
                        infos[i]["terminated"] = bool(terminations[i])
                        infos[i]["truncated"] = bool(truncations[i])

            return obs, rewards, dones, infos

        return out