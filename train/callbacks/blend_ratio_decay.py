from stable_baselines3.common.callbacks import BaseCallback

class BlendScheduleCallback(BaseCallback):
    def __init__(self, initial_blend: float, final_blend: float, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.initial_blend = initial_blend
        self.final_blend = final_blend
        self.total_timesteps = max(1, total_timesteps)

    def _on_step(self) -> bool:
        """ Decay the blend ratio over the total number of steps expected during training."""
        progress = min(self.model.num_timesteps / self.total_timesteps, 1.0)
        blend = self.initial_blend + progress * (self.final_blend - self.initial_blend)
        self.model.policy.blend_ratio = blend
        return True