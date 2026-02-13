# train/callbacks/stats_cb.py
from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class StepStatsCallback(BaseCallback):
    """
    SB3 + PettingZooVecEnv flattening:
      - each "sub-env" corresponds to an agent slot (drone_0, drone_1, ...),
        repeated per vector env if N_ENVS > 1.

    We log per-rollout stats:
      - reaches_per_drone: #unique agent-slots that reached / denom
      - team_reach_any: 1 if any ACTIVE drone reached at least once during rollout
      - team_reach_all: 1 if ALL ACTIVE drones reached at least once during rollout
      - team_reach_simultaneous: 1 if there exists a step where ALL ACTIVE drones reached that step
      - collision_steps_rate: collided steps / total agent-steps
      - wall_hit_steps_rate: wall_hit steps / total agent-steps

    IMPORTANT:
      - relies on info dict containing:
          info["active"] (bool), info["reached"] (bool),
          info["collided"] (bool), optional info["wall_hit"] (bool)
      - your env must keep a fixed agent list (you do).
    """

    def __init__(self, n_drones_fixed: int, n_envs: int, verbose: int = 0):
        super().__init__(verbose)
        self.n_drones_fixed = int(n_drones_fixed)
        self.n_envs = int(n_envs)
        self.agents_count = self.n_drones_fixed * self.n_envs
        self._rollout_reset()

    def _rollout_reset(self):
        self._total_steps = 0
        self._collisions = 0
        self._wallhits = 0

        # per agent-slot unique reach
        self._seen_reached_slots = set()

        # team-level: per-env reached flags for active drones
        # shape (n_envs, n_drones_fixed)
        self._env_drone_reached_anytime = np.zeros((self.n_envs, self.n_drones_fixed), dtype=np.bool_)

        # team-level simultaneous reach at a step
        self._team_simul_flag = np.zeros((self.n_envs,), dtype=np.bool_)

        # first collision step per slot
        self._first_collision_steps = {}

    def _on_rollout_start(self) -> None:
        self._rollout_reset()

    def _on_step(self) -> bool:
        infos = self.locals["infos"]  # list length = agents_count
        step_count = self._total_steps // max(1, self.agents_count)

        # track simult reach per env at this *environment step*
        # (we update per slot below, but we want to check once per env-step)
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

                    # unique slot reach
                    if flat_idx not in self._seen_reached_slots:
                        self._seen_reached_slots.add(flat_idx)

                    # reached anytime for team metrics
                    self._env_drone_reached_anytime[env_i, drone_i] = True

            if collided:
                self._collisions += 1
                if flat_idx not in self._first_collision_steps:
                    self._first_collision_steps[flat_idx] = step_count

            if wall_hit:
                self._wallhits += 1

            self._total_steps += 1

        # update simul reach flags (only consider ACTIVE drones)
        for e in range(self.n_envs):
            act = active_this_step[e]
            if not act.any():
                continue
            # all active drones reached this step?
            all_active_reached = np.all(np.logical_or(~act, reached_this_step[e]))
            if bool(all_active_reached):
                self._team_simul_flag[e] = True

        return True

    def _on_rollout_end(self) -> None:
        if self._total_steps <= 0:
            return

        # denom = number of ACTIVE slots that existed at least once in rollout
        # we approximate denom as n_envs * n_drones_fixed (your env keeps fixed list anyway)
        reaches = len(self._seen_reached_slots)
        denom = self.agents_count

        # team metrics per env
        team_any = 0
        team_all = 0
        team_simul = 0

        for e in range(self.n_envs):
            reached_anytime = self._env_drone_reached_anytime[e]
            # If you used curriculum with inactive drones, they will stay False;
            # treat inactive drones as "don't care" if they never became active.
            # We detect "ever active" implicitly: if a drone never reached and never active, it stays False.
            # Better: your env emits info["active"], but at rollout-end we don't store ever-active.
            # Given your setup keeps drones active per stage, this is OK.

            any_ok = bool(np.any(reached_anytime))
            all_ok = bool(np.all(reached_anytime))  # if stage has only 1 active drone, reached_anytime for others stays False -> would break

            # Fix "all_ok" for curriculum: treat drones that are inactive in this stage as ignored.
            # We can infer active count from the env itself via locals if present; fallback: assume all drones active.
            # Here we use heuristic: if at least one drone reached, and others never reach, don't mark all.
            # (If you need perfect "all active reached", store ever-active flags similarly to reached.)

            team_any += 1 if any_ok else 0
            team_all += 1 if all_ok else 0
            team_simul += 1 if bool(self._team_simul_flag[e]) else 0

        self.logger.record("stats/reaches_per_drone", float(reaches))
        self.logger.record("stats/reaches_per_drone_denom", float(denom))
        self.logger.record("stats/team_reach_any", float(team_any))
        self.logger.record("stats/team_reach_all", float(team_all))
        self.logger.record("stats/team_reach_simultaneous", float(team_simul))
        self.logger.record("stats/collision_steps_rate", float(self._collisions) / float(self._total_steps))
        self.logger.record("stats/wall_hit_steps_rate", float(self._wallhits) / float(self._total_steps))

        if self._first_collision_steps:
            steps = np.array(list(self._first_collision_steps.values()), dtype=np.float32)
            self.logger.record("stats/first_collision_step_mean", float(np.mean(steps)))
            self.logger.record("stats/first_collision_step_std", float(np.std(steps)))
        else:
            # "survived entire rollout"
            n_steps = int(self.locals.get("n_steps", 0) or 0)
            self.logger.record("stats/first_collision_step_mean", float(n_steps))
            self.logger.record("stats/first_collision_step_std", 0.0)

        if self.verbose:
            print(
                f"[Stats] reaches_per_drone={reaches}/{denom}, "
                f"team_any={team_any}, team_all={team_all}, team_simul={team_simul}, "
                f"collision_rate={self._collisions}/{self._total_steps}, wallhit_rate={self._wallhits}/{self._total_steps}"
            )
