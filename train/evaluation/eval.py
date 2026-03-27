import os
import re
from typing import Optional

import numpy as np
from pettingzoo import ParallelEnv
from stable_baselines3 import SAC
from tqdm import trange
from tabulate import tabulate

from config.defaults import (
    MODELS_DIR
)

from wrappers.env_3d import DroneSwarm3D


class EvaluateTraining:

    def __init__(
            self,
            n_drones: int,
            n_static_obstacles: int,
            n_dynamic_obstacles: int,
            lidar: bool,
            model_num: int,
            render_mode: Optional[str] = None,
            world_scale: float = 5.0,
            max_steps: int = 1000,
            num_episodes: int = 1000,
            compute_path_optimality: bool = False, # set True if your env emits 'optimal_path_length'
    ):

        self.n_drones = n_drones
        self.n_static_obstacles = n_static_obstacles
        self.n_dynamic_obstacles = n_dynamic_obstacles
        self.max_steps = max_steps
        self.lidar = lidar
        self.render_mode = render_mode
        self.world_scale = world_scale
        self.num_episodes = num_episodes
        self.compute_path_optimality = compute_path_optimality

        self.MODEL_N = model_num

        self.MODEL_PREFIX = "sac"

        self.seed = None

        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        self.MODEL_NAME = f"{self.MODEL_PREFIX}_{self.MODEL_N}_{self.n_drones}_{self.n_static_obstacles}_{self.n_dynamic_obstacles}"
        self.MODEL_PATH = os.path.join(PROJECT_ROOT, MODELS_DIR, f"{self.MODEL_NAME}")

        self.collision_free = []
        self.goal_reached = []
        self.avg_goal_reached = []
        self.avg_collision = []
        self.episode_lengths = []
        self.wall_hits_count = []
        self.path_opt_ratios = [] if self.compute_path_optimality else None
        self.bad_seeds = []

        self.env = DroneSwarm3D(
            world_scale=self.world_scale,
            num_drones_per_swarm=self.n_drones,
            num_static_obstacles=self.n_static_obstacles,
            num_dynamic_obstacles=self.n_dynamic_obstacles,
            lidar=self.lidar,
            max_steps=self.max_steps,
            render_mode=render_mode,
        )

        # thresholds for “problematic” episodes
        self.COLLISION_LOGGING_THRESHOLD = 0  # any collision
        self.WALL_HITS_LOGGING_THRESHOLD = 0  # any wall hit

        # where to dump “bad seeds”
        self.OUTPUT_DIR = "./bad_episodes_videos"
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)


    def evaluate_model(self):
        model = SAC.load(self.MODEL_PATH)
        rng = np.random.default_rng()

        eval_num_drones, eval_num_static_obstacles, eval_num_dynamic_obstacles = (
            self.n_drones, self.n_static_obstacles, self.n_dynamic_obstacles)

        print(f"Evaluating Model at location: {self.MODEL_PATH}")
        print(
            f"Env config: drones={eval_num_drones} static_obstacles={eval_num_static_obstacles} dynamic_obstacles={eval_num_dynamic_obstacles} "
            f"max_steps={self.max_steps} lidar={self.lidar} render={self.render_mode}\n"
        )

        for _ep_idx in trange(self.num_episodes, desc="Evaluating episodes"):
            self.seed = int(rng.integers(0, 2**31))

            env = self.env
            obs_dict, _ = env.reset(seed=self.seed)
            ep_len = 0

            if self.compute_path_optimality:
                pos_hist = {a: [] for a in env.possible_agents}
                opt_lengths = []

            done = {a: False for a in env.possible_agents}
            trunc = {a: False for a in env.possible_agents}

            while True:
                # ---------- per-agent prediction ----------
                act_dict = {
                    a: model.predict(obs, deterministic=True)[0]
                    for a, obs in obs_dict.items()
                }

                obs_dict, _rew_dict, done, trunc, info = env.step(act_dict)
                ep_len += 1

                # ---------- optional path optimality logging ----------
                if self.compute_path_optimality:
                    for a in env.possible_agents:
                        pos_hist[a].append(info[a].get("position"))
                    if "optimal_path_length" in info:
                        opt_lengths.append(info["optimal_path_length"])

                # ---------- terminal condition ----------
                if all(done[a] or trunc[a] for a in done):
                    sample_agent = next(iter(info))
                    stats = info[sample_agent]

                    coll_count = stats["collided"]
                    goals = stats["reached"]
                    wall_hits = stats.get("wall_collision", 0)

                    self.collision_free.append(coll_count == 0)
                    all_reached = goals == eval_num_drones
                    self.goal_reached.append(all_reached)
                    self.avg_goal_reached.append(goals / eval_num_drones)
                    self.avg_collision.append(coll_count)
                    self.wall_hits_count.append(wall_hits)
                    self.episode_lengths.append(ep_len)

                    # ---------- mark bad seeds (we only store seeds, not videos) ----------
                    if (
                        coll_count > self.COLLISION_LOGGING_THRESHOLD
                        or wall_hits > self.WALL_HITS_LOGGING_THRESHOLD
                        or not all_reached
                    ):
                        self.bad_seeds.append(self.seed)

                    # ---------- compute path optimality ratio ----------
                    if self.compute_path_optimality and opt_lengths:
                        total_actual = 0.0
                        for hist in pos_hist.values():
                            for p0, p1 in zip(hist, hist[1:]):
                                if p0 is None or p1 is None:
                                    continue
                                total_actual += np.linalg.norm(np.array(p1) - np.array(p0))
                        self.path_opt_ratios.append(
                            np.mean(opt_lengths) / total_actual if total_actual > 0 else 0.0
                        )

                    break

            env.close()

        # ---------- write out bad seeds so you can replay them ----------
        bad_log_path = f"{self.OUTPUT_DIR}/bad_episodes_{self.MODEL_NAME}.txt"
        with open(bad_log_path, "w") as f:
            for s in self.bad_seeds:
                f.write(f"{s}\n")

        # ---------- aggregated metrics ----------
        print(f"Collision-free Rate     : {np.mean(self.collision_free) * 100:6.2f}%")
        print(f"Goal Completion Rate    : {np.mean(self.goal_reached) * 100:6.2f}%")
        print(f"Avg Goal Reached        : {np.mean(self.avg_goal_reached):6.2f}")
        print(f"Avg Collisions          : {np.mean(self.avg_collision):6.2f}")
        print(f"Avg Episode Length      : {np.mean(self.episode_lengths):6.2f}")
        print(f"Avg Wall Hits/Episode   : {np.mean(self.wall_hits_count):6.2f}")
        print(f"Bad seeds written to    : {bad_log_path}")

        results_path = f"{self.OUTPUT_DIR}/results_{self.MODEL_NAME}.txt"
        with open(results_path, "w") as f:
            f.write(f"Results for the model: {self.MODEL_PATH}\n\n")
            f.write(f"Collision-free Rate     : {np.mean(self.collision_free) * 100:6.2f}%\n")
            f.write(f"Goal Completion Rate    : {np.mean(self.goal_reached) * 100:6.2f}%\n")
            f.write(f"Avg Goal Reached        : {np.mean(self.avg_goal_reached):6.2f}\n")
            f.write(f"Avg Collisions          : {np.mean(self.avg_collision):6.2f}\n")
            f.write(f"Avg Episode Length      : {np.mean(self.episode_lengths):6.2f}\n")
            f.write(f"Avg Wall Hits/Episode   : {np.mean(self.wall_hits_count):6.2f}\n")

        if self.compute_path_optimality:
            print(f"Avg Path Optimality     : {np.mean(self.path_opt_ratios):6.4f}")


    def replay_episode(self):
        model = SAC.load(self.MODEL_PATH)

        env = self.env
        obs_dict, _ = env.reset(seed=self.seed)

        while True:
            act_dict = {a: model.predict(obs, deterministic=True)[0]
                        for a, obs in obs_dict.items()}
            obs_dict, _, done, trunc, info = env.step(act_dict)

            env.render()

            if all(done[a] or trunc[a] for a in done):
                sample_agent = next(iter(info))
                stats = info[sample_agent]
                coll_count = stats["collision_count"]
                goals = stats["goals_reached"]
                wall_hits = stats.get("wall_hits", 0)

                table_data = [
                    ["Metric", "Value"],
                    ["Collisions", coll_count],
                    ["Goals Reached", goals],
                    ["Wall Hits", wall_hits],
                ]
                print(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))
                break

        env.close()


    def replay_bad_logs(self, model_n: int):
        """
        Replay a previous bad-seed file for a given model_n using current config's drones/obstacles.
        (Kept for convenience; it's not required.)
        """
        model_name = self.MODEL_NAME
        log_path = f"{self.OUTPUT_DIR}/bad_episodes_{model_name}.txt"

        with open(log_path, "r") as f:
            for line in f:
                seed = int(line.strip())
                print(f"Running seed: {seed}")
                self.replay_episode()


if __name__ == "__main__":
    model_number = 0

    evaluation = EvaluateTraining(
        model_num = model_number,
        n_drones= 1,
        n_static_obstacles= 0,
        n_dynamic_obstacles= 0,
        lidar= True,
        render_mode= None,
    )
    evaluation.evaluate_model()

    # evaluation.replay_episode(2)
    # evaluation.replay_bad_logs(39)






