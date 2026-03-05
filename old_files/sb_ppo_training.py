# train/sb_ppo_training.py
"""SB3 PPO training entrypoint (3D PyBullet drones).

All defaults are imported from config/experiment.py
Curriculum is imported from config/curriculum.py

Run:
  python -m train.sb_ppo_training
"""

from __future__ import annotations

import os
import time

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from config.curriculum import PPO_CURRICULUM
from old_files.experiment import (
    DEFAULT_LIDAR,
    DEFAULT_SHARED_TARGET,
    DEFAULT_SWARM_MAX_SEP,
    DEFAULT_SWARM_MIN_SEP,
    DEFAULT_TERMINATE_ON_COHESION_BREAK,
    DEFAULT_TERMINATE_ON_SEPARATION,
    LIDAR_HORIZONTAL_FOV_DEG,
    LIDAR_NOISE_STD,
    LIDAR_NUM_LAYERS,
    LIDAR_NUM_RAYS,
    LIDAR_RANGE,
    LIDAR_SHOW_RAYS_TRAIN,
    LIDAR_VERTICAL_FOV_DEG,
    MODELS_DIR,
    DEFAULT_NUM_DRONES,
)
from train.callbacks.stats_cb import StepStatsCallback, CurriculumStageInfo
from wrappers.pz_parallel_3d import env as make_env
from wrappers.sb3_reset_fix import SB3VecResetTupleFix

# =============================================================================
# OUTPUT PATHS
# =============================================================================
MODEL_DIR = MODELS_DIR  # keep identical behavior, just sourced from config

# =============================================================================
# PPO HYPERPARAMETERS (unchanged)
# =============================================================================
N_STEPS = 512           #TODO: TUNE SO DRONES CAN REACH TARGET
N_ENVS = 1
GAMMA = 0.99

BATCH_SIZE = 256
N_EPOCHS = 6

FIXED_NET_ARCH = dict(pi=[512, 256, 128], vf=[512, 256, 128])
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.25
ENTROPY = 0.008

# =============================================================================
# ENV SETTINGS (now sourced from config)
# =============================================================================
LIDAR = DEFAULT_LIDAR

# IMPORTANT: keep constant across entire SB3 training lifetime
# (SB3 + vectorized env expects fixed shapes/agent counts)
NUM_DRONES_FIXED = DEFAULT_NUM_DRONES

SHARED_TARGET = DEFAULT_SHARED_TARGET
SWARM_MAX_SEP = DEFAULT_SWARM_MAX_SEP
SWARM_MIN_SEP = DEFAULT_SWARM_MIN_SEP

TERMINATE_ON_COHESION_BREAK = DEFAULT_TERMINATE_ON_COHESION_BREAK
TERMINATE_ON_SEPARATION = DEFAULT_TERMINATE_ON_SEPARATION


# =============================================================================
# FILESYSTEM HELPERS
# =============================================================================
def get_next_model_path(
    base_path: str = f"{MODEL_DIR}/ppo_",
    drones: int = 1,
    obstacles: int = 0,
    ext: str = ".zip",
) -> str:
    """Find a new model filename that does not overwrite an existing model."""
    k = 0
    while os.path.isfile(f"{base_path}{k}_{drones}_{obstacles}{ext}"):
        k += 1
    return f"{base_path}{k}_{drones}_{obstacles}{ext}"


def get_next_checkpoint_dir(
    base_path: str = f"{MODEL_DIR}/ppo_drones_",
    n_drones: int = 1,
    n_obs: int = 0,
) -> str:
    """Find a new checkpoint directory name that does not overwrite an existing directory."""
    k = 0
    while os.path.isdir(f"{base_path}{k}_{n_drones}_{n_obs}"):
        k += 1
    return f"ppo_drones_{k}_{n_drones}_{n_obs}"


# =============================================================================
# ENV FACTORY (KEEP PURE: same inputs -> same env)
# =============================================================================
def create_env(num_drones: int, active_drones: int, num_obstacles: int, world_scale: float):
    """Build the PettingZoo ParallelEnv and adapt it to SB3 VecEnv."""

    # ---------- Base PettingZoo env ----------
    raw = make_env(
        num_drones=num_drones,          # constant across stages
        active_drones=active_drones,    # curriculum knob
        num_obstacles=num_obstacles,
        max_steps=N_STEPS,
        lidar=LIDAR,
        render_mode=None,
        world_scale=world_scale,
        obstacle_dynamic=True,
        debug=False,

        # swarm + shared target
        shared_target=SHARED_TARGET,
        swarm_max_sep=SWARM_MAX_SEP,
        swarm_min_sep=SWARM_MIN_SEP,
        terminate_on_cohesion_break=TERMINATE_ON_COHESION_BREAK,
        terminate_on_separation=TERMINATE_ON_SEPARATION,

        # lidar config
        lidar_num_rays=LIDAR_NUM_RAYS,
        lidar_num_layers=LIDAR_NUM_LAYERS,
        lidar_range=LIDAR_RANGE,
        lidar_vertical_fov_deg=LIDAR_VERTICAL_FOV_DEG,
        lidar_horizontal_fov_deg=LIDAR_HORIZONTAL_FOV_DEG,
        lidar_noise_std=LIDAR_NOISE_STD,
        lidar_show_rays=LIDAR_SHOW_RAYS_TRAIN,
    )

    # ---------- PettingZoo -> SB3 VecEnv ----------
    vec = ss.pettingzoo_env_to_vec_env_v1(raw)

    if N_ENVS > 1:
        vec = ss.concat_vec_envs_v1(vec, N_ENVS, base_class="stable_baselines3")

    # Fixes reset()/step() tuple mismatch between Gymnasium/SuperSuit/SB3
    vec = SB3VecResetTupleFix(vec)

    # Observation normalization (shared across curriculum stages)
    vec = VecNormalize(
        vec,
        gamma=GAMMA,
        norm_reward=False,
        norm_obs=True,
        clip_obs=5.0,
        clip_reward=10.0,
    )

    return vec


# =============================================================================
# TRAINING LOOP
# =============================================================================
def train(device: str = "auto"):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ---------- Device selection ----------
    if device == "auto":
        try:
            import torch
            device_ = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device_ = "cpu"
    else:
        device_ = device

    print(f"Using {device_} device")

    # ---------- Curriculum ----------
    curr = PPO_CURRICULUM
    total_planned_timesteps = int(sum(int(stage[3]) for stage in curr))

    # Build stage-0 env first
    world_scale0, active_drones0, obs0, _ = curr[0]
    print(
        f"\n=== Stage 0 | world={world_scale0} active={active_drones0}/{NUM_DRONES_FIXED} "
        f"obs={obs0} | shared_target={SHARED_TARGET} swarm_max={SWARM_MAX_SEP} swarm_min={SWARM_MIN_SEP} | "
        f"terminate_far={TERMINATE_ON_COHESION_BREAK} terminate_close={TERMINATE_ON_SEPARATION} | "
        f"lidar={LIDAR} rays={LIDAR_NUM_RAYS} layers={LIDAR_NUM_LAYERS} ==="
    )
    train_env = create_env(NUM_DRONES_FIXED, active_drones0, obs0, world_scale0)

    # ---------- SB3 PPO model ----------
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        n_epochs=N_EPOCHS,
        policy_kwargs=dict(net_arch=FIXED_NET_ARCH),
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENTROPY,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=None,
        verbose=1,
        device=device_,
    )

    # ---------- Curriculum stages ----------
    for stage_idx, (world_scale, active_drones, num_obstacles, stage_steps) in enumerate(curr):
        stage_steps = int(stage_steps)

        print(
            f"\n=== Stage {stage_idx} | world={world_scale} active={active_drones}/{NUM_DRONES_FIXED} "
            f"obs={num_obstacles} steps={stage_steps} | shared_target={SHARED_TARGET} "
            f"swarm_max={SWARM_MAX_SEP} swarm_min={SWARM_MIN_SEP} | "
            f"terminate_far={TERMINATE_ON_COHESION_BREAK} terminate_close={TERMINATE_ON_SEPARATION} | "
            f"lidar={LIDAR} rays={LIDAR_NUM_RAYS} layers={LIDAR_NUM_LAYERS} ==="
        )

        old_env = model.get_env()
        new_env = create_env(NUM_DRONES_FIXED, active_drones, num_obstacles, world_scale)

        # Share normalization stats between stages
        try:
            if isinstance(old_env, VecNormalize) and isinstance(new_env, VecNormalize):
                new_env.obs_rms = old_env.obs_rms
                new_env.ret_rms = old_env.ret_rms
                new_env.training = True
        except Exception:
            pass

        model.set_env(new_env)

        # ---------- Callbacks ----------
        checkpoint_dir = get_next_checkpoint_dir(n_drones=NUM_DRONES_FIXED, n_obs=num_obstacles)
        checkpoint_cb = CheckpointCallback(
            save_freq=max(1, 20_000 // max(1, N_ENVS)),
            save_path=f"{MODEL_DIR}/{checkpoint_dir}",
            name_prefix="ppo_",
        )

        # Curriculum info for stats callback (prints + TB logs)
        stage_info = CurriculumStageInfo(
            stage_idx=int(stage_idx),
            world_scale=float(world_scale),
            active_drones=int(active_drones),
            num_obstacles=int(num_obstacles),
            stage_timesteps=int(stage_steps),
            total_planned_timesteps=int(total_planned_timesteps),
        )

        stats_cb = StepStatsCallback(
            n_drones_fixed=NUM_DRONES_FIXED,
            n_envs=N_ENVS,
            stage_info=stage_info,
            verbose=1,
        )

        callbacks = CallbackList([stats_cb, checkpoint_cb])

        # ---------- Train this stage ----------
        model.learn(
            total_timesteps=stage_steps,
            reset_num_timesteps=False,  # keep global progress increasing across stages
            callback=callbacks,
        )

        # ---------- Save model snapshot ----------
        model_path = get_next_model_path(drones=NUM_DRONES_FIXED, obstacles=num_obstacles)
        model.save(model_path)
        print(f"[SAVED] {model_path}")

        # Close previous env (avoid PyBullet piling up)
        try:
            train_env.close()
        except Exception:
            pass
        train_env = new_env

    train_env.close()


if __name__ == "__main__":
    t0 = time.time()
    train(device="auto")
    print(f"Training completed in {time.time() - t0:.2f} seconds")
