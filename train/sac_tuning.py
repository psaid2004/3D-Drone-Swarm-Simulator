from __future__ import annotations
import optuna
import torch
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from config.sac_defaults import *
from config.curriculum import Curriculum
from train.sac_training import *
# reuse your existing functions
# create_env, unwrap_vec_normalize, safe_close, Curriculum, etc.


def sample_sac_params(trial):
    """Define search space (VERY IMPORTANT)"""

    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 3e-4),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "gamma": trial.suggest_categorical("gamma", [0.99, 0.995]),
        "tau": trial.suggest_categorical("tau", [0.003, 0.005, 0.01]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 2, 4]),
        "net_arch": trial.suggest_categorical("net_arch", [[256, 256], [400, 300]]),
        "ent_coef": trial.suggest_categorical("ent_coef", ["auto", "auto_0.1"]),
    }


def objective(trial):
    """One training run = one Optuna trial"""
    params = sample_sac_params(trial)

    print(f"\n=== Starting trial {trial.number} ===")
    print(f"Testing params: {params}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    params = sample_sac_params(trial)

    # ---- USE ONLY STAGE 0 ----
    curr = Curriculum().stages
    ws0, nd0, ns0, ndyn0, steps0, horizon0, lidar0 = curr[0]

    # shorten training for tuning
    TUNING_TIMESTEPS = 50000

    train_env = create_env(
        world_scale=ws0,
        num_drones=nd0,
        num_static_obstacles=ns0,
        num_dynamic_obstacles=ndyn0,
        max_episode_steps=horizon0,
        lidar_enabled=lidar0,
    )

    eval_env = create_env(
        world_scale=ws0,
        num_drones=nd0,
        num_static_obstacles=ns0,
        num_dynamic_obstacles=ndyn0,
        max_episode_steps=horizon0,
        lidar_enabled=lidar0,
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        gamma=params["gamma"],
        tau=params["tau"],
        gradient_steps=params["gradient_steps"],
        ent_coef=params["ent_coef"],
        policy_kwargs=dict(net_arch=params["net_arch"]),
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        train_freq=TRAIN_FREQ,
        verbose=0,
        device=device,
    )

    try:
        model.learn(total_timesteps=TUNING_TIMESTEPS)

        mean_reward, _ = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=5,
            deterministic=True
        )

    except Exception as e:
        print(f"Trial failed: {e}")
        mean_reward = -1e9  # penalize failure

    finally:
        safe_close(train_env)
        safe_close(eval_env)

    return mean_reward

def run_tuning():
    study = optuna.create_study(direction="maximize")

    study.optimize(
        objective,
        n_trials=30,     # increase later (50–100)
        n_jobs=1,        # set >1 for parallel
        show_progress_bar = True
    )

    print("\n===== BEST PARAMS =====")
    print(study.best_params)
    print("Best reward:", study.best_value)

    return study.best_params


if __name__ == "__main__":
    best_params = run_tuning()