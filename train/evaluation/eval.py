import os
import re

import numpy as np
from pettingzoo import ParallelEnv
from stable_baselines3 import PPO
from tqdm import trange
from tabulate import tabulate

from old_files.experiment import (
    DEFAULT_NUM_DRONES,
    DEFAULT_NUM_OBSTACLES,
    DEFAULT_MAX_STEPS,
    DEFAULT_LIDAR,
    DEFAULT_RENDER_MODE,
    MODEL_PREFIX,
    MODELS_DIR,
)

# =============================================================================
# 3D ONLY: PettingZoo + PyBullet wrapper
# =============================================================================
from wrappers.pz_parallel_3d import env as make_env

# =============================================================================
# Global configuration (centralized defaults; can override locally if needed)
# =============================================================================
NUM_DRONES = DEFAULT_NUM_DRONES
NUM_OBSTACLES = DEFAULT_NUM_OBSTACLES
MAX_STEPS = DEFAULT_MAX_STEPS
LIDAR = DEFAULT_LIDAR
RENDER_MODE = DEFAULT_RENDER_MODE

N_EVAL_EPISODES = 1000
COMPUTE_PATH_OPTIMALITY = False  # set True if your env emits 'optimal_path_length'

# thresholds for “problematic” episodes
COLLISION_LOGGING_THRESHOLD = 0  # any collision
WALL_HITS_LOGGING_THRESHOLD = 0  # any wall hit

# where to dump “bad seeds”
OUTPUT_DIR = "./bad_episodes_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Model selection
# =============================================================================
MODEL_N = 2

# naming: ppo_{MODEL_N}_{NUM_DRONES}_{NUM_OBSTACLES}.zip
MODEL_NAME = f"{MODEL_PREFIX}{MODEL_N}_{NUM_DRONES}_{NUM_OBSTACLES}"
MODEL_PATH = os.path.join("..", MODELS_DIR, f"{MODEL_NAME}.zip")


# =============================================================================
# Helpers
# =============================================================================
def infer_env_from_model_name(model_name: str):
    """
    Parses naming convention:
      ppo_<k>_<drones>_<obstacles>
    Returns (drones, obstacles) or (None, None) if it doesn't match.
    """
    m = re.match(rf"^{re.escape(MODEL_PREFIX)}\d+_(\d+)_(\d+)$", model_name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _make_eval_env(n_drones: int, n_obstacles: int, render_mode):
    """
    Adapter for the 3D wrapper env signature.
    IMPORTANT: This assumes wrappers/pz_parallel_3d.env accepts positional args:
      (num_drones, num_obstacles, max_steps, lidar, render_mode=...)
    If your 3D env factory uses keyword-only args, update this accordingly.
    """
    env: ParallelEnv = make_env(
        n_drones, n_obstacles, MAX_STEPS, LIDAR, render_mode=render_mode
    )
    return env


def evaluate_model():
    model = PPO.load(MODEL_PATH)
    rng = np.random.default_rng()

    collision_free = []
    goal_reached = []
    avg_goal_reached = []
    avg_collision = []
    episode_lengths = []
    wall_hits_count = []
    path_opt_ratios = [] if COMPUTE_PATH_OPTIMALITY else None
    bad_seeds = []

    # ---------- ensure env params match model name ----------
    inferred = infer_env_from_model_name(MODEL_NAME)
    eval_num_drones, eval_num_obstacles = NUM_DRONES, NUM_OBSTACLES
    if inferred != (None, None):
        eval_num_drones, eval_num_obstacles = inferred

    print(f"Evaluating Model at location: {MODEL_PATH}")
    print(
        f"Env config: drones={eval_num_drones} obstacles={eval_num_obstacles} "
        f"max_steps={MAX_STEPS} lidar={LIDAR} render={RENDER_MODE}\n"
    )

    for _ep_idx in trange(N_EVAL_EPISODES, desc="Evaluating episodes"):
        seed = int(rng.integers(0, 2**31))

        env = _make_eval_env(eval_num_drones, eval_num_obstacles, render_mode=RENDER_MODE)
        obs_dict, _ = env.reset(seed=seed)
        ep_len = 0

        if COMPUTE_PATH_OPTIMALITY:
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
            if COMPUTE_PATH_OPTIMALITY:
                for a in env.possible_agents:
                    pos_hist[a].append(info[a].get("position"))
                if "optimal_path_length" in info:
                    opt_lengths.append(info["optimal_path_length"])

            # ---------- terminal condition ----------
            if all(done[a] or trunc[a] for a in done):
                sample_agent = next(iter(info))
                stats = info[sample_agent]

                coll_count = stats["collision_count"]
                goals = stats["goals_reached"]
                wall_hits = stats.get("wall_hits", 0)

                collision_free.append(coll_count == 0)
                all_reached = goals == eval_num_drones
                goal_reached.append(all_reached)
                avg_goal_reached.append(goals / eval_num_drones)
                avg_collision.append(coll_count)
                wall_hits_count.append(wall_hits)
                episode_lengths.append(ep_len)

                # ---------- mark bad seeds (we only store seeds, not videos) ----------
                if (
                    coll_count > COLLISION_LOGGING_THRESHOLD
                    or wall_hits > WALL_HITS_LOGGING_THRESHOLD
                    or not all_reached
                ):
                    bad_seeds.append(seed)

                # ---------- compute path optimality ratio ----------
                if COMPUTE_PATH_OPTIMALITY and opt_lengths:
                    total_actual = 0.0
                    for hist in pos_hist.values():
                        for p0, p1 in zip(hist, hist[1:]):
                            if p0 is None or p1 is None:
                                continue
                            total_actual += np.linalg.norm(np.array(p1) - np.array(p0))
                    path_opt_ratios.append(
                        np.mean(opt_lengths) / total_actual if total_actual > 0 else 0.0
                    )

                break

        env.close()

    # ---------- write out bad seeds so you can replay them ----------
    bad_log_path = f"{OUTPUT_DIR}/bad_episodes_{MODEL_NAME}.txt"
    with open(bad_log_path, "w") as f:
        for s in bad_seeds:
            f.write(f"{s}\n")

    # ---------- aggregated metrics ----------
    print(f"Collision-free Rate     : {np.mean(collision_free) * 100:6.2f}%")
    print(f"Goal Completion Rate    : {np.mean(goal_reached) * 100:6.2f}%")
    print(f"Avg Goal Reached        : {np.mean(avg_goal_reached):6.2f}")
    print(f"Avg Collisions          : {np.mean(avg_collision):6.2f}")
    print(f"Avg Episode Length      : {np.mean(episode_lengths):6.2f}")
    print(f"Avg Wall Hits/Episode   : {np.mean(wall_hits_count):6.2f}")
    print(f"Bad seeds written to    : {bad_log_path}")

    results_path = f"{OUTPUT_DIR}/results_{MODEL_NAME}.txt"
    with open(results_path, "w") as f:
        f.write(f"Results for the model: {MODEL_PATH}\n\n")
        f.write(f"Collision-free Rate     : {np.mean(collision_free) * 100:6.2f}%\n")
        f.write(f"Goal Completion Rate    : {np.mean(goal_reached) * 100:6.2f}%\n")
        f.write(f"Avg Goal Reached        : {np.mean(avg_goal_reached):6.2f}\n")
        f.write(f"Avg Collisions          : {np.mean(avg_collision):6.2f}\n")
        f.write(f"Avg Episode Length      : {np.mean(episode_lengths):6.2f}\n")
        f.write(f"Avg Wall Hits/Episode   : {np.mean(wall_hits_count):6.2f}\n")

    if COMPUTE_PATH_OPTIMALITY:
        print(f"Avg Path Optimality     : {np.mean(path_opt_ratios):6.4f}")


def replay_episode(seed: int):
    model = PPO.load(MODEL_PATH)

    inferred = infer_env_from_model_name(MODEL_NAME)
    eval_num_drones, eval_num_obstacles = NUM_DRONES, NUM_OBSTACLES
    if inferred != (None, None):
        eval_num_drones, eval_num_obstacles = inferred

    env = _make_eval_env(eval_num_drones, eval_num_obstacles, render_mode="human")
    obs_dict, _ = env.reset(seed=seed)

    done = {a: False for a in env.possible_agents}
    trunc = done.copy()

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


def replay_bad_logs(model_n: int):
    """
    Replay a previous bad-seed file for a given model_n using current config's drones/obstacles.
    (Kept for convenience; it's not required.)
    """
    model_name = f"{MODEL_PREFIX}{model_n}_{NUM_DRONES}_{NUM_OBSTACLES}"
    log_path = f"{OUTPUT_DIR}/bad_episodes_{model_name}.txt"

    with open(log_path, "r") as f:
        for line in f:
            seed = int(line.strip())
            print(f"Running seed: {seed}")
            replay_episode(seed)


if __name__ == "__main__":
    evaluate_model()
    # replay_bad_logs(39)
    # replay_episode(1012868196)
