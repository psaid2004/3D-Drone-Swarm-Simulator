import os
import time

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from train.callbacks.stats_cb import StepStatsCallback
from wrappers.pz_parallel import env as make_env
from staged_curriculums import CURRICULUM

MODEL_DIR = "models/"
NUM_DRONES = 2
NUM_OBSTACLES = 0

# Hyperparams Configuration
N_STEPS = 512
N_ENVS = 16
GAMMA = 0.99
TRAINING_TIMESTAMPS = 1_200_000
BATCH_SIZE = 2048
# BATCH_SIZE = N_ENVS * N_STEPS
FIXED_NET_ARCH = dict(
    # pi=[256, 128, 64],
    pi=[512, 256, 128],
    vf=[512, 256, 128],
    # pi=[756, 512, 256],
    # vf=[756, 512, 256],
)
GAE_Lambda = 0.95
CLIP_RANGE = 0.25
ENTROPY = 0.008
N_EPOCHS = 6


def get_next_model_path(base_path="models/ppo_", drones=NUM_DRONES, obstacles=NUM_OBSTACLES, ext=".zip"):
    k = 0
    while os.path.isfile(f"{base_path}{k}_{drones}_{obstacles}{ext}"):
        k += 1
    return f"{base_path}{k}_{drones}_{obstacles}{ext}"


def get_next_checkpoint_dir(base_path="models/ppo_drones_", n_drones=NUM_DRONES, n_obs=NUM_OBSTACLES, ext=".zip"):
    k = 0
    while os.path.isdir(f"{base_path}{k}_{n_drones}_{n_obs}"):
        k += 1
    return f"ppo_drones_{k}_{n_drones}_{n_obs}"


def create_env(num_drones, num_obstacles, spawn_limit=100.0):
    # 1. base PettingZoo parallel env
    raw = make_env(num_drones=num_drones, num_obstacles=num_obstacles,
                   max_steps=N_STEPS, lidar=True, render_mode=None, spawn_area=spawn_limit)
    raw.render_mode = None
    # 2. keep “dead” agents alive
    wrapped = ss.black_death_v3(raw)
    # wrapped = raw
    # wrapped = raw
    # 3. SB3 VecEnv (still a single copy)
    vec = ss.pettingzoo_env_to_vec_env_v1(wrapped)

    # 4. concatenate N copies for throughput
    vec = ss.concat_vec_envs_v1(vec, N_ENVS, base_class="stable_baselines3")

    vec = VecMonitor(vec)
    # 5. normalise obs & rewards after concat
    vec = VecNormalize(vec, gamma=GAMMA, norm_reward=False, norm_obs=True, clip_obs=1.0, clip_reward=3.0)
    #
    # vec = InfoStashWrapper(vec)
    # 6. record episode stats
    return vec


def make_eval_raw(num_drones=NUM_DRONES, num_obstacles=NUM_OBSTACLES):
    return make_env(
        num_drones=num_drones,
        num_obstacles=num_obstacles,
        max_steps=N_STEPS,
        lidar=True
    )


def make_eval_env(train_norm_wrapper, num_drones=1, num_obstacles=0):
    raw = make_env(num_drones=num_drones, num_obstacles=num_obstacles,
                   max_steps=N_STEPS, lidar=False, render_mode=None)
    raw.render_mode = None

    wrapped = ss.black_death_v3(raw)
    # wrapped = raw
    vec = ss.pettingzoo_env_to_vec_env_v1(wrapped)

    vec = ss.concat_vec_envs_v1(vec, 1, base_class="stable_baselines3")

    # freeze normalisation but share statistics
    eval_norm = VecNormalize(vec, gamma=GAMMA,
                             norm_reward=True, norm_obs=True, clip_obs=10.0,
                             training=False)
    eval_norm.obs_rms = train_norm_wrapper.obs_rms
    eval_norm.ret_rms = train_norm_wrapper.ret_rms

    return VecMonitor(eval_norm)


def train(num_iters=1, curriculum_drones=2):
    for k in range(num_iters):
        train_env = create_env(num_drones=NUM_DRONES, num_obstacles=NUM_OBSTACLES, spawn_limit=100.0)
        model = PPO(
            "MlpPolicy",
            train_env,
            # learning_rate=get_linear_fn(3e-4, 1e-5, 1.0),
            learning_rate=3e-4,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            n_epochs=N_EPOCHS,
            # policy_kwargs=dict(net_arch=[128, 64, 64]),
            policy_kwargs=dict(net_arch=FIXED_NET_ARCH),
            gae_lambda=GAE_Lambda,
            clip_range=CLIP_RANGE,
            # ent_coef=0.01,
            ent_coef=ENTROPY,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=None,
            tensorboard_log="models/tb_logs/",
            verbose=1,
            device="cuda",
        )
        # print(">>> actual parallel envs:", train_env.num_envs)
        curr = CURRICULUM[curriculum_drones]
        # for stage, (spawn, drones, obs, steps) in enumerate([
        #     # (100, 2, 0, 2_000_000),
        #     # (40, 2, 0, 1_000_000),
        #     # (40, 2, 1, 700_000),
        #     # (40, 2, 2, 700_000),
        #     # (40, 2, 3, 700_000),
        #     (60, 2, 10, 2_000_000),
        #
        #     # (100, 1, 0, 2_000_000),
        #     # (40, 1, 3, 700_000),
        #     # (40, 1, 5, 700_000),
        #     # (60, 1, 8, 700_000),
        #     # (100, 1, 10, 2_000_000),
        #
        #     # (60, 2, 6, 2_000_000),
        #     # (90, 2, 8, 2_000_000),
        #     # (100, 2, 10, 3_500_000),
        #     # (100, 2, 15, 3_500_000),
        #     # (40, 2, 4, 1_200_000),
        #     # (100, 2, 6, 1_200_000),
        #     # (100, 2, 10, 2_000_000)
        # ]):
        for stage, (spawn, drones, obs, steps) in enumerate(curr):
            # if stage != 4:
            #     continue
            print(f"Stage {stage} stared...")
            old_env = model.get_env()
            training_env = create_env(num_drones=drones, num_obstacles=obs, spawn_limit=spawn)
            if isinstance(old_env, VecNormalize):
                training_env.obs_rms = old_env.obs_rms
                training_env.ret_rms = old_env.ret_rms
                training_env.training = old_env.training  # keep updating stats if you like
            model.set_env(train_env)
            # kpi_cb = KPIRecorder(verbose=1)
            checkpoint_dir = get_next_checkpoint_dir(n_drones=drones, n_obs=obs)
            # callbacks = CallbackList([kpi_cb])
            checkpoint_cb = CheckpointCallback(
                save_freq=10_000 // N_ENVS,
                save_path=f"models/{checkpoint_dir}",
                name_prefix="ppo_"
            )

            stats_cb = StepStatsCallback(drones, N_ENVS, verbose=1)
            # callbacks = CallbackList([stats_cb])
            callbacks = CallbackList([stats_cb, checkpoint_cb])

            model_path = get_next_model_path(drones=drones, obstacles=obs)
            # model.learn(total_timesteps=TRAINING_TIMESTAMPS, callback=callbacks)
            # stage_1 = int(TRAINING_TIMESTAMPS)
            # stage_1 = int(TRAINING_TIMESTAMPS / 10) + 50_000'
            if stage == 0:
                model.learn(total_timesteps=steps, progress_bar=True, callback=callbacks)
            else:
                model.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=True, callback=callbacks)
            # model.batch_size = 180
            # model.learning_rate = 3e-4
            # model.n_epochs = 16
            # model.clip_range = constant_fn(0.25)
            # model.ent_coef = 0.005
            # # train_env_danger = create_env(
            # #     num_drones=NUM_DRONES,
            # #     num_obstacles=NUM_OBSTACLES,
            # #     spawn_limit=30.0,
            # # )
            # stage_2 = (TRAINING_TIMESTAMPS // 2)
            # model.learn(total_timesteps=stage_2, reset_num_timesteps=False, progress_bar=True, callback=callbacks)
            #
            # model.set_env(train_env_danger)
            # model.learn(total_timesteps=stage_2, reset_num_timesteps=False, progress_bar=True,
            #             callback=callbacks)
            model.save(model_path)

        train_env.close()


if __name__ == "__main__":
    stage_start = time.time()
    train(num_iters=1)
    stage_duration = time.time() - stage_start
    print(f"Training completed in {stage_duration:.2f} seconds")
    # train_i(0)
    # train_staged()
