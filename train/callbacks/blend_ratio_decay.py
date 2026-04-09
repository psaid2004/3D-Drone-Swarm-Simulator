# -*- coding: utf-8 -*-
from stable_baselines3.common.callbacks import BaseCallback
from config.defaults import *
from train.callbacks.custom_stats_cb import StepStatsCallback


class BlendScheduleCallback(BaseCallback):

    def __init__(
        self,
        stats_cb: StepStatsCallback,
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
        self.stats_cb = stats_cb
    def _on_step(self) -> bool:
        steps_into_stage = self.stats_cb._stage_active_steps
        progress = min(steps_into_stage / self.stage_timesteps, 1.0)

        blend = self.blend_start + progress * (self.blend_end - self.blend_start)
        self.model.policy.blend_ratio = float(blend)
        return True