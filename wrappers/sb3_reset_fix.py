# wrappers/sb3_reset_fix.py
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper



class SB3VecResetTupleFix(VecEnvWrapper):
    """
    Fixes Gymnasium/SuperSuit incompatibilities with SB3 and makes behavior consistent.

    1) reset() may return (obs, info) -> SB3 expects obs
       - we store info in self.last_reset_info

    2) step_wait() may return:
       - SB3 style: (obs, rewards, dones, infos)
       - Gymnasium style: (obs, rewards, terminations, truncations, infos)
         -> convert to dones = terminations OR truncations
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



    def step_wait(self) -> Tuple[Any, Any, Any, Any]:
        out = self.venv.step_wait()

        if isinstance(out, tuple) and len(out) == 5:
            obs, rewards, terminations, truncations, infos = out
            dones = np.logical_or(terminations, truncations)
            try:
                dones = dones.astype(np.bool_)
            except Exception:
                pass
            return obs, rewards, dones, infos

        return out
