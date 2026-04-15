# -*- coding: utf-8 -*-
import os
from typing import Optional, List, Dict, Any

import numpy as np
from stable_baselines3 import SAC
from tqdm import trange
from tabulate import tabulate

from config.defaults import MODELS_DIR
from config.curriculum import Curriculum
from train.sac_training_with_expert import ExpertBlendPolicy
from train.expert_blender import ExpertPolicy
from wrappers.env_3d import DroneSwarm3D

"""
Evaluate trained model on all curriculum stages.
"""

class EvaluateFinalModel:
    """
    Load ONE fixed final model (the same kind your current code loads),
    and test that SAME model on ALL curriculum stages.
    """

    def __init__(
        self,
        final_model_num: int,
        final_model_n_drones: int,
        final_model_n_static_obstacles: int,
        final_model_n_dynamic_obstacles: int,
        final_model_lidar: bool,
        render_mode: Optional[str] = None,
        num_episodes: int = 30,
        compute_path_optimality: bool = False,
    ):
        self.final_model_num = final_model_num
        self.final_model_n_drones = final_model_n_drones
        self.final_model_n_static_obstacles = final_model_n_static_obstacles
        self.final_model_n_dynamic_obstacles = final_model_n_dynamic_obstacles
        self.final_model_lidar = final_model_lidar

        self.render_mode = render_mode
        self.num_episodes = num_episodes
        self.compute_path_optimality = compute_path_optimality

        self.MODEL_PREFIX = "sac"
        self.seed = None

        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        self.final_model_name = (
            f"{self.MODEL_PREFIX}_{self.final_model_num}_"
            f"{self.final_model_n_drones}_"
            f"{self.final_model_n_static_obstacles}_"
            f"{self.final_model_n_dynamic_obstacles}"
        )
        self.final_model_path = os.path.join(
            project_root,
            MODELS_DIR,
            self.final_model_name,
        )

        self.curriculum = Curriculum()

        self.output_dir = "./bad_episodes_videos"
        os.makedirs(self.output_dir, exist_ok=True)

        self.all_stage_results: List[Dict[str, Any]] = []

    def _load_fixed_final_model(self, world_scale_for_expert: float):
        """
        Always load the SAME final model.
        Only the runtime expert attachment is adapted to the stage settings.
        """
        model = SAC.load(
            self.final_model_path,
            print_system_info=True,
        )

        return model

    def _make_env(
        self,
        world_scale: float,
        n_drones: int,
        n_static_obstacles: int,
        n_dynamic_obstacles: int,
        lidar: bool,
        max_steps: int,
    ):
        return DroneSwarm3D(
            world_scale=world_scale,
            num_drones_per_swarm=n_drones,
            num_static_obstacles=n_static_obstacles,
            num_dynamic_obstacles=n_dynamic_obstacles,
            lidar=lidar,
            max_steps=max_steps,
            render_mode=self.render_mode,
        )

    def _evaluate_one_stage(
        self,
        stage_idx: int,
        world_scale: float,
        n_drones: int,
        n_static_obstacles: int,
        n_dynamic_obstacles: int,
        max_steps: int,
        lidar: bool,
    ) -> Dict[str, Any]:
        model = self._load_fixed_final_model(world_scale_for_expert=world_scale)

        env = self._make_env(
            world_scale=world_scale,
            n_drones=n_drones,
            n_static_obstacles=n_static_obstacles,
            n_dynamic_obstacles=n_dynamic_obstacles,
            lidar=lidar,
            max_steps=max_steps,
        )

        collision_free = []
        goal_reached = []
        avg_goal_reached = []
        avg_collision = []
        episode_lengths = []
        wall_hits_count = []
        path_opt_ratios = [] if self.compute_path_optimality else None
        bad_seeds = []

        rng = np.random.default_rng()

        print("\n" + "=" * 100)
        print(f"Testing fixed final model on curriculum stage {stage_idx}")
        print(f"Loaded model: {self.final_model_path}")
        print(
            f"Stage env   : world={world_scale} "
            f"drones={n_drones} "
            f"static_obstacles={n_static_obstacles} "
            f"dynamic_obstacles={n_dynamic_obstacles} "
            f"max_steps={max_steps} "
            f"lidar={lidar} "
            f"render={self.render_mode}"
        )
        print("=" * 100)

        try:
            for _ in trange(self.num_episodes, desc=f"Stage {stage_idx} episodes"):
                self.seed = int(rng.integers(0, 2**31))

                obs_dict, _ = env.reset(seed=self.seed)
                ep_len = 0

                done = {a: False for a in env.possible_agents}
                trunc = {a: False for a in env.possible_agents}

                while True:
                    agents = list(obs_dict.keys())
                    obs_batch = np.stack([obs_dict[a] for a in agents], axis=0)
                    act_batch, _ = model.predict(obs_batch, deterministic=True)

                    act_dict = {
                        agent: act_batch[i]
                        for i, agent in enumerate(agents)
                    }

                    obs_dict, _rew_dict, done, trunc, info = env.step(act_dict)
                    ep_len += 1

                    if self.render_mode is not None:
                        env.render()

                    if all(done[a] or trunc[a] for a in done):
                        sample_agent = next(iter(info))
                        stats = info[sample_agent]

                        collided = bool(stats["collided"])
                        reached = bool(stats["reached"])
                        wall_collision = bool(stats.get("wall_collision", False))

                        collision_free.append(not collided)
                        goal_reached.append(reached)
                        avg_goal_reached.append(float(reached))
                        avg_collision.append(float(collided))
                        wall_hits_count.append(float(wall_collision))
                        episode_lengths.append(ep_len)

                        if self.compute_path_optimality:
                            actual_len = stats.get("actual_path_length", None)
                            optimal_len = stats.get("optimal_path_length", None)
                            if (
                                actual_len is not None
                                and optimal_len is not None
                                and actual_len > 0
                            ):
                                path_opt_ratios.append(optimal_len / actual_len)

                        if collided or wall_collision or not reached:
                            bad_seeds.append(self.seed)

                        break
        finally:
            env.close()

        collision_free_rate = float(np.mean(collision_free)) if collision_free else 0.0
        success_rate = float(np.mean(goal_reached)) if goal_reached else 0.0
        collision_episode_rate = float(np.mean(avg_collision)) if avg_collision else 0.0
        avg_episode_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
        wall_hit_episode_rate = float(np.mean(wall_hits_count)) if wall_hits_count else 0.0
        avg_path_optimality = (
            float(np.mean(path_opt_ratios))
            if self.compute_path_optimality and path_opt_ratios
            else None
        )

        stage_tag = (
            f"stage_{stage_idx}_world_{world_scale}_drones_{n_drones}"
            f"_static_{n_static_obstacles}_dynamic_{n_dynamic_obstacles}"
        )

        bad_log_path = os.path.join(
            self.output_dir,
            f"bad_episodes_{self.final_model_name}_{stage_tag}.txt",
        )
        with open(bad_log_path, "w") as f:
            for s in bad_seeds:
                f.write(f"{s}\n")

        result = {
            "stage_idx": stage_idx,
            "world_scale": world_scale,
            "n_drones": n_drones,
            "n_static_obstacles": n_static_obstacles,
            "n_dynamic_obstacles": n_dynamic_obstacles,
            "max_steps": max_steps,
            "lidar": lidar,
            "collision_free_rate": collision_free_rate,
            "success_rate": success_rate,
            "collision_episode_rate": collision_episode_rate,
            "avg_episode_length": avg_episode_length,
            "wall_hit_episode_rate": wall_hit_episode_rate,
            "bad_seeds_count": len(bad_seeds),
            "bad_log_path": bad_log_path,
            "avg_path_optimality": avg_path_optimality,
        }

        print(f"Collision-free Rate     : {collision_free_rate * 100:6.2f}%")
        print(f"Success Rate            : {success_rate * 100:6.2f}%")
        print(f"Collision Episode Rate  : {collision_episode_rate * 100:6.2f}%")
        print(f"Avg Episode Length      : {avg_episode_length:6.2f}")
        print(f"Wall-hit Episode Rate   : {wall_hit_episode_rate * 100:6.2f}%")
        if avg_path_optimality is not None:
            print(f"Avg Path Optimality     : {avg_path_optimality:6.4f}")
        print(f"Bad seeds written to    : {bad_log_path}")

        return result

    def evaluate_all_stages(self):
        if not (
            os.path.exists(self.final_model_path)
            or os.path.exists(self.final_model_path + ".zip")
        ):
            raise FileNotFoundError(
                f"Final model not found at:\n{self.final_model_path}\n"
                f"Also checked:\n{self.final_model_path + '.zip'}"
            )

        self.all_stage_results = []

        for stage_idx, stage in enumerate(self.curriculum.stages):
            world_scale, n_drones, n_static, n_dynamic, _train_steps, episode_horizon, lidar = stage

            stage_result = self._evaluate_one_stage(
                stage_idx=stage_idx,
                world_scale=world_scale,
                n_drones=n_drones,
                n_static_obstacles=n_static,
                n_dynamic_obstacles=n_dynamic,
                max_steps=episode_horizon,
                lidar=lidar,
            )
            self.all_stage_results.append(stage_result)

        self._print_and_save_summary()

    def _print_and_save_summary(self):
        summary_table = []
        for r in self.all_stage_results:
            summary_table.append(
                [
                    r["stage_idx"],
                    r["world_scale"],
                    r["n_drones"],
                    r["n_static_obstacles"],
                    r["n_dynamic_obstacles"],
                    r["max_steps"],
                    r["lidar"],
                    f"{r['collision_free_rate'] * 100:.2f}%",
                    f"{r['success_rate'] * 100:.2f}%",
                    f"{r['collision_episode_rate'] * 100:.2f}%",
                    f"{r['wall_hit_episode_rate'] * 100:.2f}%",
                    f"{r['avg_episode_length']:.2f}",
                    r["bad_seeds_count"],
                ]
            )

        headers = [
            "Stage",
            "World",
            "Drones",
            "Static",
            "Dynamic",
            "Max Steps",
            "LiDAR",
            "Collision-Free",
            "Success",
            "Collision Ep.",
            "Wall-hit Ep.",
            "Avg Ep. Len",
            "Bad Seeds",
        ]

        print("\n" + "#" * 120)
        print(f"FIXED FINAL MODEL SUMMARY: {self.final_model_path}")
        print(tabulate(summary_table, headers=headers, tablefmt="fancy_grid"))
        print("#" * 120)

        summary_path = os.path.join(
            self.output_dir,
            f"final_model_curriculum_summary_{self.final_model_name}.txt",
        )

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Fixed final model evaluated on all curriculum stages\n")
            f.write(f"Loaded model path: {self.final_model_path}\n")
            f.write(f"Episodes per stage: {self.num_episodes}\n\n")
            f.write(tabulate(summary_table, headers=headers, tablefmt="grid"))
            f.write("\n\nDetailed per-stage results:\n\n")

            for r in self.all_stage_results:
                f.write(f"Stage {r['stage_idx']}\n")
                f.write(f"  world_scale           : {r['world_scale']}\n")
                f.write(f"  n_drones              : {r['n_drones']}\n")
                f.write(f"  n_static_obstacles    : {r['n_static_obstacles']}\n")
                f.write(f"  n_dynamic_obstacles   : {r['n_dynamic_obstacles']}\n")
                f.write(f"  max_steps             : {r['max_steps']}\n")
                f.write(f"  lidar                 : {r['lidar']}\n")
                f.write(f"  collision_free_rate   : {r['collision_free_rate'] * 100:.2f}%\n")
                f.write(f"  success_rate          : {r['success_rate'] * 100:.2f}%\n")
                f.write(f"  collision_episode_rate: {r['collision_episode_rate'] * 100:.2f}%\n")
                f.write(f"  wall_hit_episode_rate : {r['wall_hit_episode_rate'] * 100:.2f}%\n")
                f.write(f"  avg_episode_length    : {r['avg_episode_length']:.2f}\n")
                if r["avg_path_optimality"] is not None:
                    f.write(f"  avg_path_optimality   : {r['avg_path_optimality']:.4f}\n")
                f.write(f"  bad_seeds_count       : {r['bad_seeds_count']}\n")
                f.write(f"  bad_seeds_file        : {r['bad_log_path']}\n")
                f.write("\n")

        print(f"Summary written to      : {summary_path}")

    def replay_stage_episode(
        self,
        stage_idx: int,
        seed: int,
        render_mode: str = "human",
    ):
        if stage_idx < 0 or stage_idx >= len(self.curriculum.stages):
            raise ValueError(f"Invalid stage_idx {stage_idx}")

        world_scale, n_drones, n_static, n_dynamic, _train_steps, episode_horizon, lidar = self.curriculum.stages[stage_idx]

        model = self._load_fixed_final_model(world_scale_for_expert=world_scale)

        env = DroneSwarm3D(
            world_scale=world_scale,
            num_drones_per_swarm=n_drones,
            num_static_obstacles=n_static,
            num_dynamic_obstacles=n_dynamic,
            lidar=lidar,
            max_steps=episode_horizon,
            render_mode=render_mode,
        )

        try:
            obs_dict, _ = env.reset(seed=seed)

            while True:
                agents = list(obs_dict.keys())
                obs_batch = np.stack([obs_dict[a] for a in agents], axis=0)
                act_batch, _ = model.predict(obs_batch, deterministic=True)

                act_dict = {
                    agent: act_batch[i]
                    for i, agent in enumerate(agents)
                }

                obs_dict, _, done, trunc, info = env.step(act_dict)

                if all(done[a] or trunc[a] for a in done):
                    sample_agent = next(iter(info))
                    stats = info[sample_agent]

                    coll_count = stats["collided"]
                    goals = stats["reached"]
                    wall_hits = stats.get("wall_collision", 0)

                    table_data = [
                        ["Metric", "Value"],
                        ["Stage", stage_idx],
                        ["Seed", seed],
                        ["World Scale", world_scale],
                        ["Drones", n_drones],
                        ["Static Obstacles", n_static],
                        ["Dynamic Obstacles", n_dynamic],
                        ["Collisions", coll_count],
                        ["Goals Reached", goals],
                        ["Wall Hits", wall_hits],
                    ]
                    print(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))
                    break
        finally:
            env.close()


if __name__ == "__main__":
    evaluator = EvaluateFinalModel(
        #FINAL MODEL BEING EVALUATED ON ALL CIRUCULUM STAGES
        final_model_num=0,
        final_model_n_drones=1,
        final_model_n_static_obstacles=1,
        final_model_n_dynamic_obstacles=0,
        final_model_lidar=True,

        render_mode=None,   # keep None for fast evaluation
        num_episodes=50,    #set to 5 for quick testing
        compute_path_optimality=False,
    )

    evaluator.evaluate_all_stages()

    # Example replay of one seed on one stage:
    # evaluator.replay_stage_episode(stage_idx=0, seed=123456789, render_mode="human")