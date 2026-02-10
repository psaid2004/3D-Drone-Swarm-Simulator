import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class StepStatsCallback(BaseCallback):
    """
    For each sub-env (flattened agent) in infos:
      - Count a collision only the first time `collided=True` in this rollout.
      - Count a reach  only the first time `reached=True`  in this rollout.
      - Count a wall-hit only the first time `wall_hit=True`.
    At the end of each rollout, log:
      stats/collision_rate = #unique collisions / total agent‐steps
      stats/reach_rate     = #unique reaches   / total agent‐steps
      stats/wallhit_rate   = #unique wallhits  / total agent‐steps
    """

    def __init__(self, n_drones: int, n_envs: int, verbose: int = 0):
        super().__init__(verbose)
        self._rollout_reset()
        self.agents_count = n_envs * n_drones
        self._first_collision_steps = {}

    def _rollout_reset(self):
        self._unique_reaches = 0
        self._collisions = 0
        self._wallhits = 0
        self._total_steps = 0
        self._seen_reached = set()
        self._first_collision_steps = {}

    def _on_rollout_start(self) -> None:
        self._rollout_reset()

    def _on_step(self) -> bool:
        step_count = self._total_steps // self.agents_count
        infos = self.locals["infos"]
        for idx, info in enumerate(infos):
            # unique reach:
            if info.get("reached", False) and idx not in self._seen_reached:
                self._seen_reached.add(idx)
                self._unique_reaches += 1

            # collisions every time:
            if info.get("collided", False):
                self._collisions += 1
                if idx not in self._first_collision_steps:
                    self._first_collision_steps[idx] = step_count

            # wall-hits every time:
            if info.get("wall_hit", False):
                self._wallhits += 1
                if idx not in self._first_collision_steps:
                    self._first_collision_steps[idx] = step_count

            self._total_steps += 1

        return True

    def _on_rollout_end(self) -> None:
        if self._total_steps > 0:
            self.logger.record("stats/unique_reaches", self._unique_reaches)
            self.logger.record("stats/collision_steps_rate", self._collisions / self._total_steps)
            self.logger.record("stats/wall_hit_steps_rate", self._wallhits / self._total_steps)
            if self.verbose:
                print(f"[Stats] unique_reaches={self._unique_reaches}/{self.agents_count}, "
                      f"collision_rate={self._collisions}/{self._total_steps}, "
                      f"wallhit_rate={self._wallhits}/{self._total_steps}")

            if self._first_collision_steps:
                steps = np.array(list(self._first_collision_steps.values()))
                self.logger.record("stats/first_collision_step_mean", float(np.mean(steps)))
                self.logger.record("stats/first_collision_step_std", float(np.std(steps)))
            else:
                # No collisions → treat as survived entire episode
                max_step = self.locals["n_steps"]  # or your known rollout length (e.g. 512)
                self.logger.record("stats/first_collision_step_mean", float(max_step))
                self.logger.record("stats/first_collision_step_std", 0.0)