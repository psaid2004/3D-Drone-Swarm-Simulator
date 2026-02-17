# train/callbacks/stats_cb.py
from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class StepStatsCallback(BaseCallback):
    """
    Logs rollout-level metrics for SB3 training when using a PettingZoo->VecEnv adapter.

    Context:
      - With SuperSuit's pettingzoo_env_to_vec_env_v1, the environment is flattened:
        each "slot" corresponds to one agent (drone_0, drone_1, ...) and is repeated per VecEnv.
      - For N_ENVS vector envs and N_DRONES_FIXED agent slots, infos is a list of length:
            agents_count = N_ENVS * N_DRONES_FIXED

    We compute these rollout metrics:
      - stats/reaches_per_drone: number of unique agent slots that reached at least once
      - stats/reaches_per_drone_denom: denominator (= agents_count)
      - stats/team_reach_any: per-vecenv count of episodes where any ACTIVE drone reached at least once
      - stats/team_reach_all: per-vecenv count where ALL drones that were EVER ACTIVE reached at least once
      - stats/team_reach_simultaneous: per-vecenv count where there exists a step where all EVER ACTIVE drones reached
      - stats/collision_steps_rate: collided agent-steps / total agent-steps
      - stats/wall_hit_steps_rate: wall_hit agent-steps / total agent-steps
      - stats/first_collision_step_mean/std: step index of first collision per slot (mean/std)

    REQUIRED info keys emitted by your env per agent-slot:
      - "active"   (bool)   : whether the drone is active (curriculum can deactivate some drones)
      - "reached"  (bool)   : whether this drone reached goal (this step or persistent)
      - "collided" (bool)   : whether this drone collided at this step
      - "wall_hit" (bool)   : optional, defaults False

    NOTE:
      - This callback ONLY affects logging (not training dynamics).
    """

    def __init__(self, n_drones_fixed: int, n_envs: int, verbose: int = 0):
        super().__init__(verbose)
        self.n_drones_fixed = int(n_drones_fixed)
        self.n_envs = int(n_envs)
        self.agents_count = self.n_drones_fixed * self.n_envs
        self._rollout_reset()

    # -------------------------------------------------------------------------
    # Rollout state
    # -------------------------------------------------------------------------
    def _rollout_reset(self):
        # total flattened agent-steps processed this rollout
        self._total_steps = 0

        # counts over flattened agent-steps
        self._collisions = 0
        self._wallhits = 0

        # slot-level: set of flat indices that have ever reached
        self._seen_reached_slots: set[int] = set()

        # per vecenv + per drone slot:
        # - ever_active: was this drone slot active at least once during rollout?
        # - reached_anytime: did this slot reach at least once during rollout?
        self._ever_active = np.zeros((self.n_envs, self.n_drones_fixed), dtype=np.bool_)
        self._reached_anytime = np.zeros((self.n_envs, self.n_drones_fixed), dtype=np.bool_)

        # per vecenv: did we ever observe a step where all ever-active drones reached that step?
        # We'll update this progressively each step.
        self._team_simul_flag = np.zeros((self.n_envs,), dtype=np.bool_)

        # first collision step index per flat slot
        self._first_collision_steps: dict[int, int] = {}

    def _on_rollout_start(self) -> None:
        self._rollout_reset()

    # -------------------------------------------------------------------------
    # Per-step processing
    # -------------------------------------------------------------------------
    def _on_step(self) -> bool:
        infos = self.locals["infos"]  # list length = agents_count

        # Convert "flattened agent-step count" -> "environment step index"
        # Each environment step produces agents_count infos.
        env_step_idx = self._total_steps // max(1, self.agents_count)

        # We'll track reach/active per env at this environment step to compute simult.
        reached_this_step = np.zeros((self.n_envs, self.n_drones_fixed), dtype=np.bool_)
        active_this_step = np.zeros((self.n_envs, self.n_drones_fixed), dtype=np.bool_)

        for flat_idx, info in enumerate(infos):
            env_i = flat_idx // self.n_drones_fixed
            drone_i = flat_idx % self.n_drones_fixed

            active = bool(info.get("active", True))
            reached = bool(info.get("reached", False))
            collided = bool(info.get("collided", False))
            wall_hit = bool(info.get("wall_hit", False))

            # ---------- ACTIVE bookkeeping ----------
            if active:
                self._ever_active[env_i, drone_i] = True
                active_this_step[env_i, drone_i] = True

            # ---------- REACHED bookkeeping ----------
            if active and reached:
                reached_this_step[env_i, drone_i] = True
                self._reached_anytime[env_i, drone_i] = True

                # unique slot reach
                if flat_idx not in self._seen_reached_slots:
                    self._seen_reached_slots.add(flat_idx)

            # ---------- Collisions / wallhits ----------
            if collided:
                self._collisions += 1
                if flat_idx not in self._first_collision_steps:
                    self._first_collision_steps[flat_idx] = int(env_step_idx)

            if wall_hit:
                self._wallhits += 1

            self._total_steps += 1

        # ---------- Simultaneous reach update ----------
        # For each env: "all ever-active drones reached this step"
        for e in range(self.n_envs):
            ever_act = self._ever_active[e]
            if not ever_act.any():
                continue

            # A drone counts as satisfied if either:
            # - it was never active, or
            # - it was active and reached this step
            all_ever_active_reached_now = np.all(np.logical_or(~ever_act, reached_this_step[e]))
            if bool(all_ever_active_reached_now):
                self._team_simul_flag[e] = True

        return True

    # -------------------------------------------------------------------------
    # Rollout end logging
    # -------------------------------------------------------------------------
    def _on_rollout_end(self) -> None:
        if self._total_steps <= 0:
            return

        reaches = len(self._seen_reached_slots)
        denom = self.agents_count

        team_any = 0
        team_all = 0
        team_simul = 0

        for e in range(self.n_envs):
            ever_act = self._ever_active[e]
            reached_any = self._reached_anytime[e]

            if not ever_act.any():
                # no active drones at all (unlikely, but safe)
                continue

            # any ACTIVE drone reached at least once
            any_ok = bool(np.any(reached_any & ever_act))

            # all EVER ACTIVE drones reached at least once
            all_ok = bool(np.all(np.logical_or(~ever_act, reached_any)))

            team_any += 1 if any_ok else 0
            team_all += 1 if all_ok else 0
            team_simul += 1 if bool(self._team_simul_flag[e]) else 0

        # ---------- Record metrics (keys unchanged) ----------
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
            # "survived entire rollout" (fallback)
            n_steps = int(self.locals.get("n_steps", 0) or 0)
            self.logger.record("stats/first_collision_step_mean", float(n_steps))
            self.logger.record("stats/first_collision_step_std", 0.0)

        if self.verbose:
            print(
                f"[Stats] reaches_per_drone={reaches}/{denom}, "
                f"team_any={team_any}, team_all={team_all}, team_simul={team_simul}, "
                f"collision_rate={self._collisions}/{self._total_steps}, "
                f"wallhit_rate={self._wallhits}/{self._total_steps}"
            )
