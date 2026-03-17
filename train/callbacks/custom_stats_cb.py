from __future__ import annotations

from typing import Optional
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from train.callbacks.config import CurriculumStageInfo

PRINT_EVERY = 5_000


class StepStatsCallback(BaseCallback):
    """
    Logs rollout-level multi-agent stats from flattened PettingZoo infos.

    Expected per-agent info keys:
        active: bool
        reached: bool
        reached_now: bool
        just_reached: bool
        collided: bool
        collision_now: bool
        wall_collision: bool
        wall_collision_now: bool
        obstacle_collision: bool
        obstacle_collision_now: bool
        drone_collision: bool
        drone_collision_now: bool
        episode_done_all_reached: bool
        swarm_too_close: bool
        swarm_too_far: bool

    Notes:
    - "reaches_per_drone" counts UNIQUE agent-slots that reached at least once during rollout.
    - "collision_steps_rate" counts collided steps across agent-slots / total agent-steps.
    """

    def __init__(
        self,
        n_drones_fixed: int,
        n_envs: int,
        stage_info: Optional[CurriculumStageInfo] = None,
        print_every: int = PRINT_EVERY,

        enable_stage_early_stop: bool = True,
        reach_threshold: float = 0.85,
        max_collision_rate: float = 0.05,
        patience: int = 10,
        min_stage_timesteps: int = 10_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.n_drones_fixed = int(n_drones_fixed)
        self.n_envs = int(n_envs)
        self.agents_count = self.n_drones_fixed * self.n_envs

        self.stage_info = stage_info
        self.print_every = int(print_every)
        self._last_print_ts = 0

        self.enable_stage_early_stop = bool(enable_stage_early_stop)
        self.reach_threshold = float(reach_threshold)
        self.max_collision_rate = float(max_collision_rate)
        self.patience = int(patience)
        self.min_stage_timesteps = int(min_stage_timesteps)

        self._consecutive_good = 0
        self._stage_start_ts = 0
        self._should_stop = False

        self._rollout_reset()

    def reset_for_new_stage(self, stage_start_ts: int, stage_info: Optional[CurriculumStageInfo] = None) -> None:
        self._consecutive_good = 0
        self._stage_start_ts = int(stage_start_ts)
        self._should_stop = False
        if stage_info is not None:
            self.stage_info = stage_info

    def set_stage_info(self, stage_info: CurriculumStageInfo) -> None:
        self.stage_info = stage_info

    def should_stop(self) -> bool:
        return self._should_stop

    def _rollout_reset(self) -> None:
        self._agent_steps = 0
        self._env_steps = 0

        self._collision_now_count = 0
        self._wall_collision_now_count = 0
        self._obstacle_collision_now_count = 0
        self._drone_collision_now_count = 0

        self._collision_ever_count = 0
        self._wall_collision_ever_count = 0
        self._obstacle_collision_ever_count = 0
        self._drone_collision_ever_count = 0

        self._reached_slots_ever = set()
        self._reached_slots_now = set()

        self._env_drone_reached_anytime = np.zeros((self.n_envs, self.n_drones_fixed), dtype=np.bool_)
        self._env_team_simul_flag = np.zeros((self.n_envs,), dtype=np.bool_)

        self._swarm_too_close_steps = 0
        self._swarm_too_far_steps = 0

        self._first_collision_env_step = {}

    def _on_rollout_start(self) -> None:
        self._rollout_reset()

    def _on_step(self) -> bool:
        if self._should_stop:
            if self.verbose >= 1:
                print(f"[StepStatsCallback] stopping training at timestep={self.num_timesteps}")
            return False

        infos = self.locals.get("infos", None)
        if infos is None:
            return True

        self._env_steps += 1
        env_step_idx = self._env_steps - 1

        reached_this_step = np.zeros((self.n_envs, self.n_drones_fixed), dtype=np.bool_)
        active_this_step = np.zeros((self.n_envs, self.n_drones_fixed), dtype=np.bool_)

        for flat_idx, info in enumerate(infos):
            env_i = flat_idx // self.n_drones_fixed
            drone_i = flat_idx % self.n_drones_fixed

            active = bool(info.get("active", True))
            reached = bool(info.get("reached", False))
            reached_now = bool(info.get("reached_now", False))
            just_reached = bool(info.get("just_reached", False))

            collided = bool(info.get("collided", False))
            collision_now = bool(info.get("collision_now", False))

            wall_collision = bool(info.get("wall_collision", False))
            wall_collision_now = bool(info.get("wall_collision_now", False))

            obstacle_collision = bool(info.get("obstacle_collision", False))
            obstacle_collision_now = bool(info.get("obstacle_collision_now", False))

            drone_collision = bool(info.get("drone_collision", False))
            drone_collision_now = bool(info.get("drone_collision_now", False))

            swarm_too_close = bool(info.get("swarm_too_close", False))
            swarm_too_far = bool(info.get("swarm_too_far", False))

            self._agent_steps += 1

            if active:
                active_this_step[env_i, drone_i] = True

            if reached:
                self._reached_slots_ever.add(flat_idx)
                self._env_drone_reached_anytime[env_i, drone_i] = True

            if reached_now or just_reached:
                reached_this_step[env_i, drone_i] = True
                self._reached_slots_now.add(flat_idx)

            if collided:
                self._collision_ever_count += 1
                if flat_idx not in self._first_collision_env_step:
                    self._first_collision_env_step[flat_idx] = env_step_idx

            if collision_now:
                self._collision_now_count += 1
                if flat_idx not in self._first_collision_env_step:
                    self._first_collision_env_step[flat_idx] = env_step_idx

            if wall_collision:
                self._wall_collision_ever_count += 1
            if wall_collision_now:
                self._wall_collision_now_count += 1

            if obstacle_collision:
                self._obstacle_collision_ever_count += 1
            if obstacle_collision_now:
                self._obstacle_collision_now_count += 1

            if drone_collision:
                self._drone_collision_ever_count += 1
            if drone_collision_now:
                self._drone_collision_now_count += 1

            if swarm_too_close:
                self._swarm_too_close_steps += 1
            if swarm_too_far:
                self._swarm_too_far_steps += 1

        for e in range(self.n_envs):
            act = active_this_step[e]
            if not act.any():
                continue
            all_active_reached_same_step = np.all(np.logical_or(~act, reached_this_step[e]))
            if all_active_reached_same_step:
                self._env_team_simul_flag[e] = True

        return True

    def _on_rollout_end(self) -> None:
        if self._agent_steps <= 0:
            return

        denom_slots = max(1, self.agents_count)
        denom_agent_steps = max(1, self._agent_steps)
        denom_env_steps = max(1, self._env_steps)

        reaches_ever = len(self._reached_slots_ever)
        reaches_now = len(self._reached_slots_now)

        collision_step_rate = self._collision_now_count / denom_agent_steps
        wall_collision_step_rate = self._wall_collision_now_count / denom_agent_steps
        obstacle_collision_step_rate = self._obstacle_collision_now_count / denom_agent_steps
        drone_collision_step_rate = self._drone_collision_now_count / denom_agent_steps

        team_any = 0
        team_all = 0
        team_simul = 0

        for e in range(self.n_envs):
            reached_anytime = self._env_drone_reached_anytime[e]
            team_any += int(np.any(reached_anytime))
            team_all += int(np.all(reached_anytime))
            team_simul += int(self._env_team_simul_flag[e])

        self.logger.record("stats/reaches_ever_count", float(reaches_ever))
        self.logger.record("stats/reaches_now_count", float(reaches_now))
        self.logger.record("stats/reaches_denom", float(denom_slots))
        self.logger.record("stats/reach_rate", float(reaches_ever) / float(denom_slots))

        self.logger.record("stats/team_reach_any_count", float(team_any))
        self.logger.record("stats/team_reach_all_count", float(team_all))
        self.logger.record("stats/team_reach_simultaneous_count", float(team_simul))

        self.logger.record("stats/collision_step_rate", float(collision_step_rate))
        self.logger.record("stats/wall_collision_step_rate", float(wall_collision_step_rate))
        self.logger.record("stats/obstacle_collision_step_rate", float(obstacle_collision_step_rate))
        self.logger.record("stats/drone_collision_step_rate", float(drone_collision_step_rate))

        self.logger.record("stats/collision_ever_count", float(self._collision_ever_count))
        self.logger.record("stats/wall_collision_ever_count", float(self._wall_collision_ever_count))
        self.logger.record("stats/obstacle_collision_ever_count", float(self._obstacle_collision_ever_count))
        self.logger.record("stats/drone_collision_ever_count", float(self._drone_collision_ever_count))

        self.logger.record("stats/swarm_too_close_rate", float(self._swarm_too_close_steps) / float(denom_agent_steps))
        self.logger.record("stats/swarm_too_far_rate", float(self._swarm_too_far_steps) / float(denom_agent_steps))

        if self._first_collision_env_step:
            steps = np.array(list(self._first_collision_env_step.values()), dtype=np.float32)
            self.logger.record("stats/first_collision_env_step_mean", float(np.mean(steps)))
            self.logger.record("stats/first_collision_env_step_std", float(np.std(steps)))
        else:
            self.logger.record("stats/first_collision_env_step_mean", 0.0)
            self.logger.record("stats/first_collision_env_step_std", 0.0)

        if self.stage_info is not None:
            si = self.stage_info
            total = max(1, si.total_planned_timesteps)

            self.logger.record("curriculum/stage_idx", float(si.stage_idx))
            self.logger.record("curriculum/world_scale", float(si.world_scale))
            self.logger.record("curriculum/num_drones", float(si.num_drones))
            self.logger.record("curriculum/num_static_obstacles", float(si.num_static_obstacles))
            self.logger.record("curriculum/num_dynamic_obstacles", float(si.num_dynamic_obstacles))
            self.logger.record("curriculum/num_obstacles", float(si.num_obstacles))
            self.logger.record("curriculum/stage_timesteps", float(si.stage_timesteps))
            self.logger.record("curriculum/progress_frac", float(self.num_timesteps) / float(total))

        if self.verbose >= 1 and (self.num_timesteps - self._last_print_ts) >= self.print_every:
            self._last_print_ts = self.num_timesteps

            prefix = ""
            if self.stage_info is not None:
                si = self.stage_info
                total = max(1, si.total_planned_timesteps)
                pct = 100.0 * self.num_timesteps / total
                prefix = (
                    f"[Curriculum] stage={si.stage_idx} "
                    f"scale={si.world_scale:.2f} "
                    f"drones={si.num_drones} "
                    f"obs={si.num_obstacles} "
                    f"stage_ts={si.stage_timesteps} "
                    f"global_ts={self.num_timesteps}/{total} ({pct:.1f}%)\n"
                )

            print(
                prefix
                + "[Stats] "
                + f"reach={reaches_ever}/{denom_slots} "
                + f"team_any={team_any} "
                + f"team_all={team_all} "
                + f"team_simul={team_simul} "
                + f"collision_rate={collision_step_rate:.4f} "
                + f"wall={wall_collision_step_rate:.4f} "
                + f"obs={obstacle_collision_step_rate:.4f} "
                + f"drone={drone_collision_step_rate:.4f}"
            )

        if not self.enable_stage_early_stop:
            return

        steps_in_stage = self.num_timesteps - self._stage_start_ts
        if steps_in_stage < self.min_stage_timesteps:
            return

        reach_rate = float(reaches_ever) / float(denom_slots)
        meets_reach = reach_rate >= self.reach_threshold
        meets_collision = collision_step_rate <= self.max_collision_rate

        if meets_reach and meets_collision:
            self._consecutive_good += 1
        else:
            self._consecutive_good = 0

        self.logger.record("curriculum/good_rollout_streak", float(self._consecutive_good))

        if self.verbose >= 1 and self._consecutive_good > 0:
            print(
                f"[StageEarlyStop] reach={reach_rate:.3f} "
                f"collision_rate={collision_step_rate:.4f} "
                f"streak={self._consecutive_good}/{self.patience}"
            )

        if self._consecutive_good >= self.patience:
            self._should_stop = True
            if self.verbose >= 1:
                print(
                    f"[StageEarlyStop] triggered at num_timesteps={self.num_timesteps} "
                    f"(stage_steps={steps_in_stage})"
                )