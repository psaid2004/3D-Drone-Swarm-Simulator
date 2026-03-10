"""
SB3 SAC training (3D PyBullet drones) with expert blending.

All required defaults and curriculum stages are imported from the config file.

Run:
  python -m train.sb3_sac_training
"""

from __future__ import annotations

import os, time
import torch
import warnings

import supersuit as ss
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback
)
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.sac.policies import SACPolicy # TODO FIGURE OUT WHETHER THIS SHOULD BE USED INSTEAD

from config.curriculum import Curriculum
from config.defaults import *
from train.callbacks.stats_cb import StepStatsCallback, CurriculumStageInfo
from wrappers.pz_parallel_3d import DroneSwarm3D
from wrappers.sb3_reset_fix import SB3VecResetTupleFix

warnings.filterwarnings("ignore", message=".*Box low's precision lowered.*")
warnings.filterwarnings("ignore", message=".*Box high's precision lowered.*")

# ---- SAC HYPERPARAMETERS ----
GAMMA           = 0.99
BATCH_SIZE      = 256
BUFFER_SIZE     = 300_000
LEARNING_STARTS = 1_000       # random exploration before training begins
TRAIN_FREQ      = 1           # env steps between gradient updates
GRADIENT_STEPS  = 1           # gradient updates per train_freq
LEARNING_RATE   = 3e-4
ENTROPY_COEF    = "auto"

NET_ARCH = dict(pi=[512, 256, 128], qf=[512, 256, 128])

N_ENVS = 1

# ---- FILESYSTEM HELPERS ----
def get_next_model_path(base=f"{MODELS_DIR}/sac_", drones=1, obs=0, ext=".zip"):
    """Find a new model filename that does not overwrite an existing model."""
    k = 0
    while os.path.isfile(f"{base}{k}_{drones}_{obs}{ext}"):
        k += 1

    return f"{base}{k}_{drones}_{obs}{ext}"


def get_next_checkpoint_dir(base=f"{MODEL_DIR}/sac_drones_", n_drones=1, n_obs=0):
    """Find a new checkpoint directory name that does not overwrite an existing directory."""
    k = 0
    while os.path.isdir(f"{base}{k}_{n_drones}_{n_obs}"):
        k += 1

    return f"sac_drones_{k}_{n_drones}_{n_obs}"


# ---- CREATE ENV ----
def create_env(num_drones: int, active_drones: int, num_obstacles: int, world_scale: float):
    """Build PettingZoo ParallelEnv → SB3 VecEnv."""

    env = DroneSwarm3D( # TODO FIX
        world_scale = world_scale,
        num_drones_per_swarm = num_drones,
        num_static_obstacles =(num_obstacles/2),
        num_dynamic_obstacles,
        max_steps,
        num_swarms: int = 1,
        lidar: bool = False,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ),

    # TODO SEPARATE REWARD FUNCTION FROM ENV WRAPPER
    # TODO FIX THE CURRICULUM TO ACCEPT DYNAMIC AND STATIC OBSTACLES
    vec = ss.pettingzoo_env_to_vec_env_v1(env)

    vec = ss.concat_vec_envs_v1(vec, N_ENVS, base_class="stable_baselines3")

    vec = SB3VecResetTupleFix(vec)

    vec = VecMonitor(vec)

    vec = VecNormalize(
        vec,
        gamma=GAMMA,
        norm_reward=False,
        norm_obs=True,
        clip_obs=5.0,
        clip_reward=10.0,
    )

    return vec


def unwrap_vecnormalize(env):
    while env is not None:
        if isinstance(env, VecNormalize):
            return env
        env = getattr(env, "venv", None)
    return None


# ---- TRAINING LOOP ----
def train(device: str = "auto"):
    os.makedirs(MODEL_DIR, exist_ok=True)

    if device == "auto":
        device_ = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_ = device
    print(f"Device: {device_}")

    curr = Curriculum().stages
    total_planned = int(sum(int(s[3]) for s in curr))

    # --- build stage-0 env ---
    ws0, ad0, ob0, _ = curr[0]
    print(f"\n=== Stage 0 | world={ws0} active={ad0}/{DEFAULT_NUM_DRONES} obs={ob0} ===")
    train_env = create_env(DEFAULT_NUM_DRONES, ad0, ob0, ws0)


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
        tensorboard_log=f"{MODEL_DIR}/tb_logs",
        verbose=1,
        device=device_,
    )

    # --- curriculum stages ---
    for idx, (world_scale, active_drones, n_obs, stage_steps) in enumerate(curr):
        stage_steps = int(stage_steps)

        print(
            f"\n=== Stage {idx} | world={world_scale} "
            f"active={active_drones}/{DEFAULT_NUM_DRONES} "
            f"obs={n_obs} steps={stage_steps} ==="
        )

        # swap env
        old_env = model.get_env()
        new_env = create_env(DEFAULT_NUM_DRONES, active_drones, n_obs, world_scale)

        old_vn = unwrap_vecnormalize(old_env)
        new_vn = unwrap_vecnormalize(new_env)

        if old_vn is not None and new_vn is not None:
            new_vn.obs_rms = old_vn.obs_rms
            new_vn.ret_rms = old_vn.ret_rms

        model.set_env(new_env)

        # callbacks
        ckpt_dir = get_next_checkpoint_dir(n_drones=DEFAULT_NUM_DRONES, n_obs=n_obs)
        ckpt_cb = CheckpointCallback(
            save_freq=max(1, 20_000 // max(1, N_ENVS)),
            save_path=f"{MODEL_DIR}/{ckpt_dir}",
            name_prefix="sac_",
        )

        stage_info = CurriculumStageInfo(
            stage_idx=idx,
            world_scale=float(world_scale),
            active_drones=int(active_drones),
            num_obstacles=int(n_obs),
            stage_timesteps=stage_steps,
            total_planned_timesteps=total_planned,
        )
        stats_cb = StepStatsCallback(
            n_drones_fixed=DEFAULT_NUM_DRONES,
            n_envs=N_ENVS,
            stage_info=stage_info,
            enable_early_stop=True,
            reach_threshold=0.85,
            max_collision_rate=0.05,
            patience=10,
            min_timesteps=10_000,
            verbose=1,
        )
        stats_cb.reset(stage_start_ts=model.num_timesteps)

        callbacks = CallbackList([stats_cb, ckpt_cb])

        # train
        model.learn(
            total_timesteps=stage_steps,
            reset_num_timesteps=False,
            callback=callbacks,
        )

        # save
        path = get_next_model_path(drones=DEFAULT_NUM_DRONES, obs=n_obs)
        model.save(path)
        print(f"[SAVED] {path}")

        try:
            train_env.close()
        except Exception:
            pass
        train_env = new_env

        old_env.close()

    train_env.close()
    print("Training complete.")


if __name__ == "__main__":
    t0 = time.time()
    train(device="auto")
    print(f"Finished in {time.time() - t0:.1f}s")
