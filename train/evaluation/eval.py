import os

import imageio
import numpy as np
from pettingzoo import ParallelEnv
from stable_baselines3 import PPO
from tqdm import trange
from tabulate import tabulate
from wrappers.pz_parallel import env as make_env

# ───────────────────────────────────────────────────────────────────────────────
# Global configuration
# ───────────────────────────────────────────────────────────────────────────────
NUM_DRONES = 2
NUM_OBSTACLES = 4
MAX_STEPS = 510
LIDAR = True
RENDER_MODE = None
N_EVAL_EPISODES = 1000
COMPUTE_PATH_OPTIMALITY = False  # set True if your env emits 'optimal_path_length'

# thresholds for “problematic” episodes
COLLISION_LOGGING_THRESHOLD = 0  # any collision
WALL_HITS_LOGGING_THRESHOLD = 0  # any wall hit

# where to dump videos
LOG_VIDEO = True
OUTPUT_DIR = "./bad_episodes_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_N = 2
MODEL = f"ppo_{str(MODEL_N)}_{NUM_DRONES}_{NUM_OBSTACLES}"
# MODEL = "ppo_drones_10_2_4/ppo__4989344_steps"
MODEL_PATH = f"../models/{MODEL}.zip"

# ───────────────────────────────────────────────────────────────────────────────

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
    print(f"Evaluating Model at location: {MODEL_PATH}\n")
    for ep_idx in trange(N_EVAL_EPISODES, desc="Evaluating episodes"):
        seed = int(rng.integers(0, 2 ** 31))
        env: ParallelEnv = make_env(
            NUM_DRONES, NUM_OBSTACLES, MAX_STEPS, LIDAR, render_mode=RENDER_MODE
        )
        obs_dict, _ = env.reset(seed=seed)
        ep_len = 0

        if COMPUTE_PATH_OPTIMALITY:
            pos_hist = {a: [] for a in env.possible_agents}
            opt_lengths = []

        done = {a: False for a in env.possible_agents}
        trunc = {a: False for a in env.possible_agents}

        while True:

            # per-agent prediction
            act_dict = {
                a: model.predict(obs, deterministic=True)[0]
                for a, obs in obs_dict.items()
            }

            obs_dict, rew_dict, done, trunc, info = env.step(act_dict)
            ep_len += 1

            if COMPUTE_PATH_OPTIMALITY:
                for a in env.possible_agents:
                    pos_hist[a].append(info[a].get("position"))
                if "optimal_path_length" in info:
                    opt_lengths.append(info["optimal_path_length"])

            # terminal condition
            if all(done[a] or trunc[a] for a in done):
                # pick any agent for shared stats
                sample_agent = next(iter(info))
                stats = info[sample_agent]

                coll_count = stats["collision_count"]
                goals = stats["goals_reached"]
                wall_hits = stats.get("wall_hits", 0)

                collision_free.append(coll_count == 0)
                all_reached = goals == NUM_DRONES
                goal_reached.append(all_reached)
                avg_goal_reached.append(goals / NUM_DRONES)
                avg_collision.append(coll_count)
                wall_hits_count.append(wall_hits)
                episode_lengths.append(ep_len)

                # decide if this episode is “bad”
                if LOG_VIDEO and (
                        coll_count > COLLISION_LOGGING_THRESHOLD or wall_hits > WALL_HITS_LOGGING_THRESHOLD or not all_reached):
                    bad_seeds.append(seed)

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

    # write out the bad seeds so you can replay them later
    with open(f"{OUTPUT_DIR}/bad_episodes_{MODEL}.txt", "w") as f:
        for s in bad_seeds:
            f.write(f"{s}\n")

    # ────────────────────────────────────────────────────────────────────────────
    # Print aggregated metrics
    # ────────────────────────────────────────────────────────────────────────────
    print(f"Collision-free Rate     : {np.mean(collision_free) * 100:6.2f}%")
    print(f"Goal Completion Rate    : {np.mean(goal_reached) * 100:6.2f}%")
    print(f"Avg Goal Reached        : {np.mean(avg_goal_reached):6.2f}")
    print(f"Avg Collisions          : {np.mean(avg_collision):6.2f}")
    print(f"Avg Episode Length      : {np.mean(episode_lengths):6.2f}")
    print(f"Avg Wall Hits/Episode   : {np.mean(wall_hits_count):6.2f}")

    with open(f"{OUTPUT_DIR}/results_{MODEL}.txt", "w") as f:
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
    env: ParallelEnv = make_env(
        NUM_DRONES, NUM_OBSTACLES, MAX_STEPS, LIDAR, render_mode="human"
    )
    obs_dict, _ = env.reset(seed=seed)

    done = {a: False for a in env.possible_agents}
    trunc = done.copy()

    while True:
        act_dict = {a: model.predict(obs, deterministic=True)[0]
                    for a, obs in obs_dict.items()}
        obs_dict, _, done, trunc, info = env.step(act_dict)
        # print(info)
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
                ["Wall Hits", wall_hits]
            ]
            print(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))
            break

    env.close()


def write_video(ep_idx, frames):
    vid_path = os.path.join(OUTPUT_DIR, f"episode_{ep_idx:03d}.mp4")
    with imageio.get_writer(
            vid_path,
            format="FFMPEG",
            mode="I",
            fps=30,
            codec="libx264",
            ffmpeg_log_level="error"
    ) as writer:
        for img in frames:
            writer.append_data(img)
    print(f">>> Saved problematic episode {ep_idx} to {vid_path}")


def replay_bad_logs(logNumber):
    with open(f"{OUTPUT_DIR}/bad_episodes_ppo_{logNumber}_{NUM_DRONES}_{NUM_OBSTACLES}.txt", "r") as f:
        lines = f.readlines()
        for seed in lines:
            print(f"Running seed: {seed.strip()}")
            replay_episode(int(seed.strip()))


if __name__ == "__main__":
    evaluate_model()
    # replay_bad_logs(39)
    # replay_episode(1012868196)
