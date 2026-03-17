# train/callbacks/stats_cb.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

PRINT_EVERY = 5_000  # only print to console every N timesteps

@dataclass
class CurriculumStageInfo:
    stage_idx: int
    world_scale: float
    num_drones: int
    num_static_obstacles: int
    num_dynamic_obstacles: int
    stage_timesteps: int
    total_planned_timesteps: int


class StepStatsCallback(BaseCallback):
    """
    Logs and prints rollout-level stats for SB3 when using a PettingZoo ParallelEnv
    flattened through PettingZooVecEnv.

    Notes:
    - "reaches_per_drone" counts UNIQUE agent-slots that reached at least once during rollout.
    - "collision_steps_rate" counts collided steps across agent-slots / total agent-steps.
    - Uses info dict keys:
        info["active"] (bool), info["reached"] (bool),
        info["collided"] (bool), optional info["wall_hit"] (bool)
    """

    def __init__(
            self,
            n_drones_fixed: int,
            n_envs: int,
            stage_info: Optional[CurriculumStageInfo] = None,
            print_every: int = PRINT_EVERY,
            # early-stop tools
            enable_early_stop: bool = True,
            reach_threshold: float = 0.85,
            max_collision_rate: float = 0.05,
            patience: int = 10,
            min_timesteps:int = 10000,
            verbose: int = 0,
    ):
        super().__init__(verbose)
        self.n_drones_fixed = int(n_drones_fixed)
        self.n_envs = int(n_envs)
        self.agents_count = self.n_drones_fixed * self.n_envs

        self.stage_info = stage_info
        self.print_every = int(print_every)
        self._last_print_ts = 0

        self.enable_early_stop = bool(enable_early_stop)
        self.reach_threshold = float(reach_threshold)
        self.max_collision_rate = float(max_collision_rate)
        self.patience = int(patience)
        self.min_timesteps = int(min_timesteps)

        self._consecutive_good = 0
        self._stage_start_ts = 0

        self._rollout_reset()


    def reset(self, stage_start_ts: int = 0) -> None:
        self._consecutive_good = 0
        self._stage_start_ts = int(stage_start_ts)


    # You can call this from training when you enter a new curriculum stage
    def set_stage_info(self, stage_info: CurriculumStageInfo) -> None:
        self.stage_info = stage_info


    def _rollout_reset(self):
        self._total_steps = 0
        self._collisions = 0
        self._wallhits = 0
        self._seen_reached_slots = set()
        self._env_drone_reached_anytime = np.zeros((self.n_envs, self.n_drones_fixed),
                                                   dtype=np.bool_)
        self._team_simul_flag = np.zeros((self.n_envs,), dtype=np.bool_)
        self._first_collision_steps = {}


    def _on_rollout_start(self) -> None:
        self._rollout_reset()


    def _on_step(self) -> bool:
        infos = self.locals["infos"]  # list length = agents_count
        step_count = self._total_steps // max(1, self.agents_count)

        reached_this_step = np.zeros((self.n_envs, self.n_drones_fixed), dtype=np.bool_)
        active_this_step = np.zeros((self.n_envs, self.n_drones_fixed), dtype=np.bool_)

        for flat_idx, info in enumerate(infos):
            env_i = flat_idx // self.n_drones_fixed
            drone_i = flat_idx % self.n_drones_fixed

            active = bool(info.get("active", True))
            reached = bool(info.get("reached", False))
            collided = bool(info.get("collided", False))
            wall_hit = bool(info.get("wall_hit", False))

            if active:
                active_this_step[env_i, drone_i] = True
                if reached:
                    reached_this_step[env_i, drone_i] = True

                    if flat_idx not in self._seen_reached_slots:
                        self._seen_reached_slots.add(flat_idx)

                    self._env_drone_reached_anytime[env_i, drone_i] = True

            if collided:
                self._collisions += 1
                if flat_idx not in self._first_collision_steps:
                    self._first_collision_steps[flat_idx] = step_count

            if wall_hit:
                self._wallhits += 1

            self._total_steps += 1

        # simultaneous reach: all ACTIVE drones in an env reached on same env-step
        for e in range(self.n_envs):
            act = active_this_step[e]
            if not act.any():
                continue
            all_active_reached = np.all(np.logical_or(~act, reached_this_step[e]))
            if bool(all_active_reached):
                self._team_simul_flag[e] = True

        return True

    def _on_rollout_end(self) -> None:
        if self._total_steps <= 0:
            return

        reaches = len(self._seen_reached_slots)
        denom = self.agents_count
        collision_rate = float(self._collisions) / float(self._total_steps)

        team_any = 0
        team_all = 0
        team_simul = 0

        for e in range(self.n_envs):
            reached_anytime = self._env_drone_reached_anytime[e]
            any_ok = bool(np.any(reached_anytime))
            all_ok = bool(np.all(reached_anytime))
            team_any += 1 if any_ok else 0
            team_all += 1 if all_ok else 0
            team_simul += 1 if bool(self._team_simul_flag[e]) else 0


        # ---- scalar logs ----
        self.logger.record("stats/reaches_per_drone", float(reaches))
        self.logger.record("stats/reaches_per_drone_denom", float(denom))
        self.logger.record("stats/team_reach_any", float(team_any)) # how many env had every active drone reach its target
        self.logger.record("stats/team_reach_all", float(team_all)) # equal to above if single drone
        self.logger.record("stats/team_reach_simultaneous", float(team_simul))
        self.logger.record("stats/collision_steps_rate", # fraction of agent steps where a collision occurred (needs to decrease over time)
                           float(collision_rate))
        self.logger.record("stats/wall_hit_steps_rate",
                           float(self._wallhits) / float(self._total_steps))

        if self._first_collision_steps:
            steps = np.array(list(self._first_collision_steps.values()), dtype=np.float32)
            self.logger.record("stats/first_collision_step_mean", float(np.mean(steps))) # avg timestep within an episode when the first collision happens (higher = better)
            self.logger.record("stats/first_collision_step_std", float(np.std(steps)))
        else:
            self.logger.record("stats/first_collision_step_mean", 0.0)
            self.logger.record("stats/first_collision_step_std", 0.0)

        # ---- curriculum/progress logs ----
        if self.stage_info is not None:
            si = self.stage_info
            self.logger.record("curriculum/stage_idx", float(si.stage_idx))
            self.logger.record("curriculum/world_scale", float(si.world_scale))
            self.logger.record("curriculum/active_drones", float(si.active_drones))
            self.logger.record("curriculum/num_obstacles", float(si.num_obstacles))
            self.logger.record("curriculum/stage_timesteps", float(si.stage_timesteps))

            total = max(1, int(si.total_planned_timesteps))
            frac = float(self.num_timesteps) / float(total)
            self.logger.record("curriculum/progress_frac", frac)

        # ---- console print ----
        if self.verbose and (self.num_timesteps - self._last_print_ts) >= self.print_every:
            self._last_print_ts = self.num_timesteps

            prefix = ""
            if self.stage_info is not None:
                si = self.stage_info
                total = max(1, int(si.total_planned_timesteps))
                prefix = (
                    f"[Curr] stage={si.stage_idx} "
                    f"(scale={si.world_scale}, drones={si.num_drones}, "
                    f"static={si.num_static_obstacles}, dynamic={si.num_dynamic_obstacles}, "
                    f"stage_ts={si.stage_timesteps}) "
                    f"| timesteps={self.num_timesteps}/{total} ({100.0 * self.num_timesteps / total:5.1f}%)\n"
                )

            print(
                prefix
                + f"[Stats] reaches_per_drone={reaches}/{denom}, "
                  f"team_any={team_any}, team_all={team_all}, team_simul={team_simul}, "
                  f"collision_rate={self._collisions}/{self._total_steps}, wallhit_rate={self._wallhits}/{self._total_steps}"
            )

        # ---- Early stop decision ----
        if not self.enable_early_stop:
            return

        steps_in_stage = self.num_timesteps - self._stage_start_ts
        if steps_in_stage < self.min_timesteps:
            return

        reach_rate = float(reaches) / max(denom, 1.0)
        meets_reach = reach_rate >= self.reach_threshold
        meets_collision = collision_rate <= self.max_collision_rate

        if meets_reach and meets_collision:
            self._consecutive_good +=1
        else:
            self._consecutive_good = 0

        if self.verbose and self._consecutive_good > 0:
            print(
                f"  [EarlyStop] reach={reach_rate:.2f} coll={collision_rate:.4f} streak={self._consecutive_good}/{self.patience}")

        if self._consecutive_good >= self.patience:
            if self.verbose:
                print(f"  [EarlyStop] TRIGGERED at {self.num_timesteps} (stage_steps={steps_in_stage})")
            self.model._total_timesteps = self.num_timesteps  # forces learn-loop exit in practice
            self.model.env.reset()  # optional: clean reset
