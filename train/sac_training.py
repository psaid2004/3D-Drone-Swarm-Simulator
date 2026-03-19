"""
SAC training.

All required defaults and curriculum stages are imported from the config file.

Run:
  python -m train.sac_training
"""

from __future__ import annotations

import os, time
from pathlib import Path

import torch
import warnings

import supersuit as ss
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback, EvalCallback, ProgressBarCallback, StopTrainingOnNoModelImprovement

)
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor

from config.curriculum import Curriculum
from config.defaults import *
from config.sac_defaults import *

#from callbacks.custom_stats_cb import StepStatsCallback, CurriculumStageInfo
from train.callbacks.custom_stats_cb import StepStatsCallback, CurriculumStageInfo
from wrappers.env_3d import DroneSwarm3D
from wrappers.env_reset_fix import SB3VecResetTupleFix

warnings.filterwarnings("ignore", message=".*Box low's precision lowered.*")
warnings.filterwarnings("ignore", message=".*Box high's precision lowered.*")


# ---- FILESYSTEM HELPERS ----
def get_next_model_path(
        num_drones: int,
        num_static_obs: int,
        num_dynamic_obs: int,
        base=f"{MODELS_DIR}/sac_",
        ext=".zip"
):
    """Find a new model filename that does not overwrite an existing model."""
    k = 0
    while os.path.isfile(f"{base}{k}_{num_drones}_{num_static_obs}_{num_dynamic_obs}{ext}"):
        k += 1

    return f"{base}{k}_{num_drones}_{num_static_obs}_{num_dynamic_obs}{ext}"

def get_next_checkpoint_dir(
        num_drones: int,
        num_static_obs: int,
        num_dynamic_obs: int
):
    k = 0
    while os.path.isdir(f"{MODELS_DIR}/sac_drones_{k}_{num_drones}_{num_static_obs}_{num_dynamic_obs}"):
        k += 1

    return f"sac_drones_{k}_{num_drones}_{num_static_obs}_{num_dynamic_obs}"

# ---- CREATE ENV ----
def create_env(
        world_scale: float,
        num_drones: int,
        num_static_obstacles: int,
        num_dynamic_obstacles: int,
        max_episode_steps: int,
        render_mode = None
):
    """Build PettingZoo ParallelEnv → SB3 VecEnv."""

    env = DroneSwarm3D(
        world_scale = world_scale,
        num_drones_per_swarm=num_drones,
        num_static_obstacles=num_static_obstacles,
        num_dynamic_obstacles=num_dynamic_obstacles,
        max_steps=max_episode_steps,
        render_mode=render_mode,
    )

    # convert PettingZoo ParallelEnv to vectorised environment shape for SB3 to work on
    vec = ss.pettingzoo_env_to_vec_env_v1(env)
    # create number of copies of env in parallel
    vec = ss.concat_vec_envs_v1(vec, N_ENVS, base_class="stable_baselines3")

    vec = SB3VecResetTupleFix(vec)
    # record episode-level statistics from the vectorized environment
    vec = VecMonitor(vec)

    # normalizes observations and/or rewards to stabilize training
    vec = VecNormalize(
        vec,
        gamma=GAMMA,
        norm_reward=False,
        norm_obs=True,
        clip_obs=5.0,
        # clip_reward=10.0,
    )

    return vec


def unwrap_vec_normalize(env):
    while env is not None:
        if isinstance(env, VecNormalize):
            return env
        env = getattr(env, "venv", None)
    return None


def copy_vec_normalize_stats(src_env, dst_env) -> None:
    """
    Copy VecNormalize running statistics from src_env to dst_env.
    Safe no-op if either side is missing VecNormalize.
    """
    src_vn = unwrap_vec_normalize(src_env)
    dst_vn = unwrap_vec_normalize(dst_env)

    if src_vn is None or dst_vn is None:
        return

    dst_vn.obs_rms = src_vn.obs_rms
    dst_vn.ret_rms = src_vn.ret_rms


def safe_close(env) -> None:
    if env is None:
        return
    try:
        env.close()
    except Exception:
        pass


# ---- CALLBACK HELPER ----
def make_stage_callbacks(
    *,
    stage_idx: int,
    world_scale: float,
    num_drones: int,
    num_static_obs: int,
    num_dynamic_obs: int,
    stage_steps: int,
    total_planned_timesteps: int,
    model_num_timesteps: int,
    eval_env,
):
    """
    Build the full callback stack for one curriculum stage.
    """

    stage_info = CurriculumStageInfo(
        stage_idx=int(stage_idx),
        world_scale=float(world_scale),
        num_drones=int(num_drones),
        num_static_obstacles=int(num_static_obs),
        num_dynamic_obstacles=int(num_dynamic_obs),
        stage_timesteps=int(stage_steps),
        total_planned_timesteps=int(total_planned_timesteps),
    )

    ckpt_dir = get_next_checkpoint_dir(
        num_drones=num_drones,
        num_static_obs=num_static_obs,
        num_dynamic_obs=num_dynamic_obs,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 10_000 // max(1, N_ENVS)),
        save_path=f"{MODELS_DIR}/{ckpt_dir}",
        name_prefix="sac",
        save_vecnormalize=True,
        save_replay_buffer=True,
    )

    progress_cb = ProgressBarCallback()

    stats_cb = StepStatsCallback(
        num_drones=int(num_drones),
        num_envs=int(N_ENVS),
        stage_info=stage_info,
        print_every=5000,
        enable_stage_early_stop=True,
        reach_threshold=0.85,
        max_collision_rate=0.05,
        patience=10,
        min_stage_timesteps=10000,
        verbose=1,
    )

    stats_cb.reset_for_new_stage(stage_start_ts=model_num_timesteps, stage_info=stage_info)

    stop_no_improve_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=8,
        min_evals=3,
        verbose=1,
    )

    # evaluate current policy on clean evaluate environment
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(Path(MODELS_DIR) / "best_model"),
        log_path=str(Path(MODELS_DIR) / "eval_logs"),
        eval_freq=max(10_000 // max(1, N_ENVS), 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        callback_after_eval=stop_no_improve_cb,
        verbose=1,
    )

    return CallbackList([
        progress_cb,
        stats_cb,
        ckpt_cb,
        #eval_cb,
    ])


# ---- SAC TRAINING ----
def train(device: str = "auto"):
    """ Train the SAC model while looping through all curriculum stages."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    if device == "auto":
        device_ = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_ = device
    print(f"Device: {device_}")

    curr = Curriculum().stages
    total_planned_timesteps = int(sum(int(s[4]) for s in curr)) #sum of all curriculum stage timesteps

    # --- build stage-0 env ---
    ws0, nd0, ns0, ndyn0, steps0 = curr[0]
    print(
        f"\n=== Stage 0 | world={ws0} drones={nd0} "
        f"static obstacles={ns0} dynamic obstacles={ndyn0} steps={steps0} ==="
    )

    # create the training environment
    train_env = create_env(
        world_scale=ws0,
        num_drones=nd0,
        num_static_obstacles=ns0,
        num_dynamic_obstacles=ndyn0,
        max_episode_steps=steps0, # TODO IS THIS CONSIDERED THE MAX STEPS? SAME BELOW
        render_mode = "human"
    )

    # create an evaluation environment
    eval_env = create_env(
        world_scale=ws0,
        num_drones=nd0,
        num_static_obstacles=ns0,
        num_dynamic_obstacles=ndyn0,
        max_episode_steps=steps0,
    )

    # --- create SAC model ---
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        gamma=GAMMA,
        ent_coef=ENTROPY_COEF,
        tensorboard_log=f"{MODELS_DIR}/tb_logs",
        verbose=1,
        device=device_,
    ) # TODO WE NEED TO DO SOME SORT OF HYPERPARAMETER TRAINING HERE TO FIND THE OPTIMAL VALUES

    # --- train stage 0 first ---


    callbacks = make_stage_callbacks(
        stage_idx=0,
        world_scale=ws0,
        num_drones=nd0,
        num_static_obs=ns0,
        num_dynamic_obs=ndyn0,
        stage_steps=steps0,
        total_planned_timesteps=total_planned_timesteps,
        model_num_timesteps=model.num_timesteps,
        eval_env=eval_env,
    )

    model.learn(
        total_timesteps=steps0,
        reset_num_timesteps=False,
        callback=callbacks,
    )

    path = get_next_model_path(
        num_drones=nd0,
        num_static_obs=ns0,
        num_dynamic_obs=ndyn0,
    )
    model.save(path)
    print(f"[SAVED] {path}")

    vn = unwrap_vec_normalize(model.get_env())
    if vn is not None:
        norm_path = path.replace(".zip", "_vec_normalize.pkl")
        vn.save(norm_path)
        print(f"[SAVED VECNORMALIZE] {norm_path}")

    # --- curriculum stages ---
    for idx, (world_scale, num_drones, num_static_obs, num_dynamic_obs, stage_steps) in enumerate(curr[1:], start=1):
        stage_steps = int(stage_steps)

        print(
            f"\n=== Stage {idx} | world={world_scale} "
            f"drones={num_drones} static={num_static_obs} "
            f"dynamic={num_dynamic_obs} steps={stage_steps} ==="
        )

        # swap envs
        old_train_env = model.get_env()
        old_eval_env = eval_env

        new_train_env = create_env(
            world_scale=world_scale,
            num_drones=num_drones,
            num_static_obstacles=num_static_obs,
            num_dynamic_obstacles=num_dynamic_obs,
            max_episode_steps=stage_steps,
        )

        new_eval_env = create_env(
            world_scale=world_scale,
            num_drones=num_drones,
            num_static_obstacles=num_static_obs,
            num_dynamic_obstacles=num_dynamic_obs,
            max_episode_steps=stage_steps,
        )

        # Carry normalization stats forward.
        copy_vec_normalize_stats(old_train_env, new_train_env)
        copy_vec_normalize_stats(old_train_env, new_eval_env)

        model.set_env(new_train_env)
        eval_env = new_eval_env

        callbacks = make_stage_callbacks(
            stage_idx=idx,
            world_scale=world_scale,
            num_drones=num_drones,
            num_static_obs=num_static_obs,
            num_dynamic_obs=num_dynamic_obs,
            stage_steps=stage_steps,
            total_planned_timesteps=total_planned_timesteps,
            model_num_timesteps=model.num_timesteps,
            eval_env=eval_env,
        )

        model.learn(
            total_timesteps=stage_steps,
            reset_num_timesteps=False,
            callback=callbacks,
        )

        # save the model up until now
        path = get_next_model_path(
            num_drones = num_drones,
            num_static_obs = num_static_obs,
            num_dynamic_obs = num_dynamic_obs
        )

        model.save(path)
        print(f"[SAVED MODEL] {path}")

        # save the normalised observation to use during evaluation
        vn = unwrap_vec_normalize(model.get_env())
        if vn is not None:
            norm_path = path.replace(".zip", "_vec_normalize.pkl")
            vn.save(norm_path)
            print(f"[SAVED VECNORMALIZE] {norm_path}")

        safe_close(old_eval_env)
        safe_close(old_train_env)

        train_env = new_train_env

    # save the final normalised values
    final_vn = unwrap_vec_normalize(model.get_env())
    if final_vn is not None:
        final_norm_path = f"{MODELS_DIR}/final_vec_normalize.pkl"
        final_vn.save(final_norm_path)
        print(f"[SAVED FINAL VEC_NORMALIZE] {final_norm_path}")

    safe_close(eval_env)
    safe_close(train_env)

    print("Training complete.")


if __name__ == "__main__":
    t0 = time.time()
    train(device="auto")
    print(f"Finished in {time.time() - t0:.1f}s")
