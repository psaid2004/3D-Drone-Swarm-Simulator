# train/sb_ppo_training.py
import os
import time

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize

from train.callbacks.stats_cb import StepStatsCallback
from wrappers.pz_parallel_3d import env as make_env
from wrappers.sb3_reset_fix import SB3VecResetTupleFix

MODEL_DIR = "models/"

# rollout length
N_STEPS = 512
N_ENVS = 1
GAMMA = 0.99

BATCH_SIZE = 256
N_EPOCHS = 6

FIXED_NET_ARCH = dict(pi=[512, 256, 128], vf=[512, 256, 128])
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.25
ENTROPY = 0.008

# train with lidar
LIDAR = True

# IMPORTANT: keep constant across entire SB3 training lifetime
NUM_DRONES_FIXED = 2

# swarm task constraints
SWARM_MAX_SEP = 2.0
SWARM_MIN_SEP = 0.25
SHARED_TARGET = True


def get_next_model_path(base_path="models/ppo_", drones=1, obstacles=0, ext=".zip"):
    k = 0
    while os.path.isfile(f"{base_path}{k}_{drones}_{obstacles}{ext}"):
        k += 1
    return f"{base_path}{k}_{drones}_{obstacles}{ext}"


def get_next_checkpoint_dir(base_path="models/ppo_drones_", n_drones=1, n_obs=0):
    k = 0
    while os.path.isdir(f"{base_path}{k}_{n_drones}_{n_obs}"):
        k += 1
    return f"ppo_drones_{k}_{n_drones}_{n_obs}"


def create_env(num_drones: int, active_drones: int, num_obstacles: int, world_scale: float):
    raw = make_env(
        num_drones=num_drones,          # constant across all stages
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
        terminate_on_cohesion_break=False,
        terminate_on_separation=False,
        # lidar config
        lidar_num_rays=72,
        lidar_num_layers=5,
        lidar_range=6.0,
        lidar_vertical_fov_deg=45.0,
        lidar_horizontal_fov_deg=360.0,
        lidar_noise_std=0.0,
        lidar_show_rays=False,   # training: OFF
    )
    raw.render_mode = None

    vec = ss.pettingzoo_env_to_vec_env_v1(raw)

    if N_ENVS > 1:
        vec = ss.concat_vec_envs_v1(vec, N_ENVS, base_class="stable_baselines3")

    vec = SB3VecResetTupleFix(vec)

    vec = VecNormalize(
        vec,
        gamma=GAMMA,
        norm_reward=False,
        norm_obs=True,
        clip_obs=5.0,
        clip_reward=10.0,
    )

    return vec


def train(device: str = "auto"):
    os.makedirs(MODEL_DIR, exist_ok=True)

    if device == "auto":
        try:
            import torch
            device_ = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device_ = "cpu"
    else:
        device_ = device

    print(f"Using {device_} device")

    # Curriculum: (world_scale, active_drones, num_obstacles, timesteps)
    CURR = [
        (2.0, 1, 10, 200_000),
        (2.5, 1, 8,  400_000),
        (3.0, 1, 6,  500_000),
        (3.5, 2, 6,  600_000),
        (4.5, 2, 8,  800_000),
    ]

    world_scale, active_drones, obs, _ = CURR[0]
    print(
        f"\n=== Stage 0 | world={world_scale} active={active_drones}/{NUM_DRONES_FIXED} "
        f"obs={obs} | shared_target={SHARED_TARGET} swarm_max={SWARM_MAX_SEP} | "
        f"lidar={LIDAR} rays=72 layers=5 ==="
    )
    train_env = create_env(NUM_DRONES_FIXED, active_drones, obs, world_scale)

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

    for stage, (world_scale, active_drones, obs, steps) in enumerate(CURR):
        print(
            f"\n=== Stage {stage} | world={world_scale} active={active_drones}/{NUM_DRONES_FIXED} "
            f"obs={obs} steps={steps} | shared_target={SHARED_TARGET} swarm_max={SWARM_MAX_SEP} | "
            f"lidar={LIDAR} rays=72 layers=5 ==="
        )

        old_env = model.get_env()
        new_env = create_env(NUM_DRONES_FIXED, active_drones, obs, world_scale)

        # Share normalization stats between stages
        try:
            if isinstance(old_env, VecNormalize) and isinstance(new_env, VecNormalize):
                new_env.obs_rms = old_env.obs_rms
                new_env.ret_rms = old_env.ret_rms
                new_env.training = True
        except Exception:
            pass

        model.set_env(new_env)

        checkpoint_dir = get_next_checkpoint_dir(n_drones=NUM_DRONES_FIXED, n_obs=obs)
        checkpoint_cb = CheckpointCallback(
            save_freq=max(1, 20_000 // max(1, N_ENVS)),
            save_path=f"models/{checkpoint_dir}",
            name_prefix="ppo_",
        )

        stats_cb = StepStatsCallback(NUM_DRONES_FIXED, N_ENVS, verbose=1)
        callbacks = CallbackList([stats_cb, checkpoint_cb])

        model.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            callback=callbacks,
        )

        model_path = get_next_model_path(drones=NUM_DRONES_FIXED, obstacles=obs)
        model.save(model_path)
        print(f"[SAVED] {model_path}")

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
