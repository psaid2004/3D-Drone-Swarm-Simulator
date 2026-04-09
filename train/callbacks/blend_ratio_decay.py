# -*- coding: utf-8 -*-
from stable_baselines3.common.callbacks import BaseCallback


class BlendScheduleCallback(BaseCallback):

    def __init__(
        self,
        blend_start: float,
        blend_end: float,
        stage_timesteps: int,
        stage_start_ts: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.blend_start = blend_start
        self.blend_end = blend_end
        self.stage_timesteps = max(1, stage_timesteps)
        self.stage_start_ts = stage_start_ts

    def _on_step(self) -> bool:
        steps_into_stage = self.model.num_timesteps - self.stage_start_ts
        progress = min(steps_into_stage / self.stage_timesteps, 1.0)
        blend = self.blend_start + progress * (self.blend_end - self.blend_start)
        self.model.policy.blend_ratio = float(blend)
        return True