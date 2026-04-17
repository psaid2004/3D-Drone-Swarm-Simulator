# -*- coding: utf-8 -*-
import os
from typing import Optional

import numpy as np
from stable_baselines3 import SAC
from tqdm import trange
from tabulate import tabulate

from config.defaults import MODELS_DIR
from wrappers.env_3d import DroneSwarm3D

"""
Evaluate trained model on respective curriculum stage.
"""

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
        max_steps: int = 4000,
        num_episodes: int = 30,
    ):
        self.n_drones = n_drones
        self.n_static_obstacles = n_static_obstacles
        self.n_dynamic_obstacles = n_dynamic_obstacles
        self.max_steps = max_steps
        self.lidar = lidar
        self.render_mode = render_mode
        self.world_scale = world_scale
        self.num_episodes = num_episodes

        self.MODEL_N = model_num
        self.MODEL_PREFIX = "sac"
        self.seed = None

        PROJECT_ROOT = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        self.MODEL_NAME = f"{self.MODEL_PREFIX}_{self.MODEL_N}_{self.n_drones}_{self.n_static_obstacles}_{self.n_dynamic_obstacles}"
        self.MODEL_PATH = os.path.join(PROJECT_ROOT, MODELS_DIR, f"{self.MODEL_NAME}")

        self.collision_free = []
        self.goal_reached = []
        self.avg_goal_reached = []
        self.avg_collision = []
        self.episode_lengths = []
        self.wall_hits_count = []
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

    def _load_model(self):
        model = SAC.load(
            self.MODEL_PATH,
            print_system_info=True,
        )


        return model

    def evaluate_model(self):
        model = self._load_model()
        rng = np.random.default_rng()

        print(f"Evaluating model at: {self.MODEL_PATH}")
        print(
            f"Env config: drones={self.n_drones} "
            f"static_obstacles={self.n_static_obstacles} "
            f"dynamic_obstacles={self.n_dynamic_obstacles} "
            f"max_steps={self.max_steps} "
            f"lidar={self.lidar} "
            f"render={self.render_mode}\n"
        )

        try:
            for _ in trange(self.num_episodes, desc="Evaluating episodes"):
                self.seed = int(rng.integers(0, 2**31))

                obs_dict, _ = self.env.reset(seed=self.seed)
                ep_len = 0

                done = {a: False for a in self.env.possible_agents}
                trunc = {a: False for a in self.env.possible_agents}

                while True:

                    agents = list(obs_dict.keys())
                    obs_batch = np.stack([obs_dict[a] for a in agents], axis=0)
                    act_batch, _ = model.predict(obs_batch, deterministic=True)

                    act_dict = {
                        agent: act_batch[i]
                        for i, agent in enumerate(agents)
                    }

                    obs_dict, _rew_dict, done, trunc, info = self.env.step(act_dict)
                    ep_len += 1

                    if self.render_mode is not None:
                        self.env.render()

                    if all(done[a] or trunc[a] for a in done):
                        sample_agent = next(iter(info))
                        stats = info[sample_agent]

                        # keep these aligned with your env's actual keys
                        collided = bool(stats["collided"])
                        reached = bool(stats["reached"])
                        wall_collision = bool(stats.get("wall_collision", False))

                        self.collision_free.append(not collided)
                        self.goal_reached.append(reached)

                        # If reached means full episode success, this should just be 0/1
                        self.avg_goal_reached.append(float(reached))

                        self.avg_collision.append(float(collided))
                        self.wall_hits_count.append(float(wall_collision))
                        self.episode_lengths.append(ep_len)

                        if collided or wall_collision or not reached:
                            self.bad_seeds.append(self.seed)

                        break
        finally:
            self.env.close()

        bad_log_path = f"{self.OUTPUT_DIR}/bad_episodes_{self.MODEL_NAME}.txt"
        with open(bad_log_path, "w") as f:
            for s in self.bad_seeds:
                f.write(f"{s}\n")

        print(f"Collision-free Rate     : {np.mean(self.collision_free) * 100:6.2f}%")
        print(f"Success Rate            : {np.mean(self.goal_reached) * 100:6.2f}%")
        print(f"Collision Episode Rate  : {np.mean(self.avg_collision) * 100:6.2f}%")
        print(f"Avg Episode Length      : {np.mean(self.episode_lengths):6.2f}")
        print(f"Wall-hit Episode Rate   : {np.mean(self.wall_hits_count) * 100:6.2f}%")
        print(f"Bad seeds written to    : {bad_log_path}")

        results_path = f"{self.OUTPUT_DIR}/results_{self.MODEL_NAME}.txt"
        with open(results_path, "w") as f:
            f.write(f"Results for the model: {self.MODEL_PATH}\n\n")
            f.write(
                f"Collision-free Rate     : {np.mean(self.collision_free) * 100:6.2f}%\n"
            )
            f.write(
                f"Success Rate            : {np.mean(self.goal_reached) * 100:6.2f}%\n"
            )
            f.write(
                f"Collision Episode Rate  : {np.mean(self.avg_collision) * 100:6.2f}%\n"
            )
            f.write(f"Avg Episode Length      : {np.mean(self.episode_lengths):6.2f}\n")
            f.write(
                f"Wall-hit Episode Rate   : {np.mean(self.wall_hits_count) * 100:6.2f}%\n"
            )


    def replay_episode(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = seed
        if self.seed is None:
            raise ValueError(
                "No seed set. Pass seed=... or run evaluate_model() first."
            )

        model = self._load_model()

        env = DroneSwarm3D(
            world_scale=self.world_scale,
            num_drones_per_swarm=self.n_drones,
            num_static_obstacles=self.n_static_obstacles,
            num_dynamic_obstacles=self.n_dynamic_obstacles,
            lidar=self.lidar,
            max_steps=self.max_steps,
            render_mode=self.render_mode,
        )

        try:
            obs_dict, _ = env.reset(seed=self.seed)

            while True:
                act_dict = {
                    a: model.predict(obs, deterministic=True)[0]
                    for a, obs in obs_dict.items()
                }
                obs_dict, _, done, trunc, info = env.step(act_dict)

                if all(done[a] or trunc[a] for a in done):
                    sample_agent = next(iter(info))
                    stats = info[sample_agent]

                    # use same keys as evaluate_model()
                    coll_count = stats["collided"]
                    goals = stats["reached"]
                    wall_hits = stats.get("wall_collision", 0)

                    table_data = [
                        ["Metric", "Value"],
                        ["Seed", self.seed],
                        ["Collisions", coll_count],
                        ["Goals Reached", goals],
                        ["Wall Hits", wall_hits],
                    ]
                    print(
                        tabulate(table_data, headers="firstrow", tablefmt="fancy_grid")
                    )
                    break
        finally:
            env.close()

    def replay_bad_logs(self):
        log_path = f"{self.OUTPUT_DIR}/bad_episodes_{self.MODEL_NAME}.txt"

        with open(log_path, "r") as f:
            for line in f:
                self.seed = int(line.strip())
                print(f"Running seed: {self.seed}")
                self.replay_episode()


if __name__ == "__main__":
    model_number = 0

    evaluation = EvaluateTraining(
        model_num=model_number,
        n_drones=8,
        n_static_obstacles=8,
        n_dynamic_obstacles=2,
        lidar=True,
        render_mode=None, #"human",
    )
    evaluation.evaluate_model()

    # evaluation.replay_episode(seed=733460456)
    # evaluation.replay_bad_logs()
