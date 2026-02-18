"""
SB3 SAC training (3D PyBullet drones) with expert blending.

All required defaults and curriculum stages are imported from the config file.

Run:
  python -m train.sb3_sac_training
"""

from __future__ import annotations

import os
import time
import numpy as np
import torch
import supersuit as ss

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.sac.policies import SACPolicy

from config.curriculum import SAC_CURRICULUM
from config.experiment import (
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

# ---------------------------- OUTPUT PATHS -----------------------------------
MODEL_DIR = MODELS_DIR

# ---------------------------- SAC HYPERPARAMETERS ----------------------------
GAMMA = 0.99
BATCH_SIZE = 256
ENTROPY_COEF = "auto"

FIXED_NET_ARCH = dict(pi=[512, 256, 128], qf=[512, 256, 128])  # policy + Q-network

N_STEPS = 512
N_ENVS = 1


# ---------------------------- FILESYSTEM HELPERS ----------------------------
def get_next_model_path(
        base_path: str = f"{MODEL_DIR}/sac_",
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
        base_path: str = f"{MODEL_DIR}/sac_drones_",
        n_drones: int = 1,
        n_obs: int = 0,
) -> str:
    """Find a new checkpoint directory name that does not overwrite an existing directory."""
    k = 0
    while os.path.isdir(f"{base_path}{k}_{n_drones}_{n_obs}"):
        k += 1
    return f"sac_drones_{k}_{n_drones}_{n_obs}"


# ---------------------------- EXPERT POLICY ----------------------------
class ExpertPolicy:
    def __init__(self):
        self.dynamic_avoid_direction = None
        self.static_avoid_direction = None

    def expert_action(self, state):
        """
        Compute expert action based on drone state with 3D LIDAR.
        Assumes state structure:
        [ pos(3), target(3), ... , lidar(360 = 72*5) ]
        """
        # ---------------- POSITION & TARGET ----------------
        current_pos = np.array(state[:3])
        target_pos = np.array(state[3:6])
        direction_to_target = target_pos - current_pos
        direction_to_target[2] = 0  # only horizontal
        dist_to_target = np.linalg.norm(direction_to_target) + 1e-6
        direction_to_target /= dist_to_target  # normalize

        # ---------------- LIDAR PROCESSING ----------------
        lidar_raw = np.array(state[-(LIDAR_NUM_RAYS*LIDAR_NUM_LAYERS):])
        lidar = lidar_raw.reshape(LIDAR_NUM_LAYERS, LIDAR_NUM_RAYS)

        # Use middle layer for obstacle decisions
        mid_layer = LIDAR_NUM_LAYERS // 2

        # front = middle ray, left = first ray, right = last ray
        front_distance = np.mean(lidar[mid_layer, LIDAR_NUM_RAYS//2 - 1:LIDAR_NUM_RAYS//2 + 2])
        left_free_space = np.mean(lidar[mid_layer, :3])
        right_free_space = np.mean(lidar[mid_layer, -3:])

        obstacle_velocity = np.array(state[9:11])
        left_edge, right_edge = state[11], state[12]

        future_obstacle_pos = current_pos[:2] + obstacle_velocity * 0.5
        # check if the moving obstacle is in the drone's path
        obstacle_ahead = (future_obstacle_pos[0] > current_pos[0]) and (
                np.linalg.norm(future_obstacle_pos - current_pos[:2]) < 3.0)

        moving_towards_drone = np.linalg.norm(obstacle_velocity) > 0.2
        action = np.array([direction_to_target[0], direction_to_target[1], 0.0])
        current_y = state[1]
        obstacle_center = (left_edge + right_edge) / 2

        # -- Moving Obstacle Avoidance --
        if obstacle_ahead and moving_towards_drone:
            print("Moving obstacle detected in path! Adjusting route.")
            if self.moving_avoid_direction is None:  # Choose direction only once

                if right_free_space > 0.5:
                    self.moving_avoid_direction = "right"
                    print("Moving Right to Avoid Moving Obstacle.")
                elif left_free_space > 0.5:
                    self.moving_avoid_direction = "left"
                    print("Moving Left to Avoid Moving Obstacle.")
                else:
                    self.moving_avoid_direction = "stop"
                    print("No space to move! Slowing down.")

                # Cooldown to prevent rapid switching

            # Execute Moving Obstacle Avoidance
            if self.moving_avoid_direction == "right":
                action = np.array([0, -1, 0])  # Move right
            elif self.moving_avoid_direction == "left":
                action = np.array([0, 1, 0])  # Move left
            elif self.moving_avoid_direction == "stop":
                action = np.array([-0.5, 0, 0])  # Slow down

        elif front_distance < 5.5:
            if self.static_avoid_direction is None:  # Only choose once per obstacle
                if current_y < obstacle_center and right_free_space < -45:
                    self.static_avoid_direction = "right"
                elif left_free_space > 0.5:
                    self.static_avoid_direction = "left"
                else:
                    self.static_avoid_direction = "stop"

                # execute the chosen action
            if self.static_avoid_direction == "right":
                action = np.array([0, -1, 0])  # Move right
                print("Avoiding obstacle! Moving Right.")
            elif self.static_avoid_direction == "left":
                action = np.array([0, 1, 0])  # Move left
                print("Avoiding obstacle! Moving Left.")
            elif self.static_avoid_direction == "stop":
                action = np.array([-0.5, 0, 0])  # Slow down
                print("Obstacle ahead & no free space! Slowing down.")

        elif left_free_space < 0.5:
            print("Obstacle on the left! Moving straight.")
            self.static_avoid_direction = None
            self.moving_avoid_direction = None
            action = np.array([-1, 0, 0])

        elif right_free_space > -45:
            print("Obstacle on the right! Moving straight")
            self.static_avoid_direction = None
            self.moving_avoid_direction = None
            action = np.array([-1, 0, 0])
        else:
            action = np.array([direction_to_target[0], direction_to_target[1], 0.0])
            self.static_avoid_direction = None
            self.moving_avoid_direction = None
            print("target")

        return action


class ExpertBlendPolicy(SACPolicy):
    def __init__(self, *args, expert_policy=None, blend_ratio=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_policy = expert_policy
        self.blend_ratio = blend_ratio

    def forward(self, obs, deterministic=False):
        action_rl, _ = super()._predict(obs, deterministic=deterministic)

        # Convert obs to numpy for expert
        obs_np = obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs
        action_expert = np.array([self.expert_policy.expert_action(obs_np[0])])

        # Blend actions
        blended = self.blend_ratio * action_expert + (
                1 - self.blend_ratio) * action_rl.cpu().detach().numpy()
        return torch.tensor(blended, dtype=torch.float32, device=action_rl.device), None


# ---------------------------- CREATE ENV ----------------------------
def create_env(num_drones: int, active_drones: int, num_obstacles: int, world_scale: float):
    """Build the PettingZoo ParallelEnv and adapt it to SB3 VecEnv."""

    # ---------- Base PettingZoo env ----------
    raw = make_env(
        num_drones=DEFAULT_NUM_DRONES,
        active_drones=active_drones,
        num_obstacles=num_obstacles,
        max_steps=N_STEPS,
        lidar=DEFAULT_LIDAR,
        render_mode=None,
        world_scale=world_scale,
        obstacle_dynamic=True,
        debug=False,

        # swarm + shared target
        shared_target=DEFAULT_SHARED_TARGET,
        swarm_max_sep=DEFAULT_SWARM_MAX_SEP,
        swarm_min_sep=DEFAULT_SWARM_MIN_SEP,
        terminate_on_cohesion_break=DEFAULT_TERMINATE_ON_COHESION_BREAK,
        terminate_on_separation=DEFAULT_TERMINATE_ON_SEPARATION,

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


# ---------------------------- TRAINING LOOP ----------------------------
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

    expert_policy = ExpertPolicy()
    curr = SAC_CURRICULUM
    total_planned_timesteps = int(sum(int(stage[3]) for stage in curr))

    # Build stage-0 env first
    world_scale0, active_drones0, obs0, _ = curr[0]
    print(
        f"\n=== Stage 0 | world={world_scale0} active={active_drones0}/{DEFAULT_NUM_DRONES} "
        f"obs={obs0} | shared_target={DEFAULT_SHARED_TARGET} swarm_max={DEFAULT_SWARM_MAX_SEP} swarm_min={DEFAULT_SWARM_MIN_SEP} | "
        f"terminate_far={DEFAULT_TERMINATE_ON_COHESION_BREAK} terminate_close={DEFAULT_TERMINATE_ON_SEPARATION} | "
        f"lidar={DEFAULT_LIDAR} rays={LIDAR_NUM_RAYS} layers={LIDAR_NUM_LAYERS} ==="
    )
    train_env = create_env(DEFAULT_NUM_DRONES, active_drones0, obs0, world_scale0)

    # ---------- SB3 PPO model ----------
    model = SAC(
        ExpertBlendPolicy,
        train_env,
        learning_rate=3e-4,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        policy_kwargs=dict(expert_policy=expert_policy, blend_ratio=0.8),
        ent_coef=ENTROPY_COEF,
        tensorboard_log=None,
        verbose=1,
        device=device_,
    )

    # ---------- Curriculum stages ----------
    for stage_idx, (world_scale, active_drones, num_obstacles, stage_steps) in enumerate(curr):
        stage_steps = int(stage_steps)

        print(
            f"\n=== Stage {stage_idx} | world={world_scale} active={active_drones}/{DEFAULT_NUM_DRONES} "
            f"obs={num_obstacles} steps={stage_steps} | shared_target={DEFAULT_SHARED_TARGET} "
            f"swarm_max={DEFAULT_SWARM_MAX_SEP} swarm_min={DEFAULT_SWARM_MIN_SEP} | "
            f"terminate_far={DEFAULT_TERMINATE_ON_COHESION_BREAK} terminate_close={DEFAULT_TERMINATE_ON_SEPARATION} | "
            f"lidar={DEFAULT_LIDAR} rays={LIDAR_NUM_RAYS} layers={LIDAR_NUM_LAYERS} ==="
        )

        old_env = model.get_env()
        new_env = create_env(DEFAULT_NUM_DRONES, active_drones, num_obstacles, world_scale)

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
        checkpoint_dir = get_next_checkpoint_dir(n_drones=DEFAULT_NUM_DRONES, n_obs=num_obstacles)
        checkpoint_cb = CheckpointCallback(
            save_freq=max(1, 20_000 // max(1, N_ENVS)),
            save_path=f"{MODEL_DIR}/{checkpoint_dir}",
            name_prefix="sac_",
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
            n_drones_fixed=DEFAULT_NUM_DRONES,
            n_envs=N_ENVS,
            stage_info=stage_info,
            verbose=1,
        )

        callbacks = CallbackList([stats_cb, checkpoint_cb])

        # ---------- Train this stage ----------
        model.learn(
            total_timesteps=30, # change back to stage_steps
            reset_num_timesteps=False,
            callback=callbacks,
        )

        # ---------- Save model snapshot ----------
        model_path = get_next_model_path(drones=DEFAULT_NUM_DRONES, obstacles=num_obstacles)
        model.save(model_path)
        print(f"[SAVED] {model_path}")

        # Close previous env (avoid PyBullet piling up)
        try:
            train_env.close()
        except Exception:
            pass
        train_env = new_env

    train_env.close()

# def debug_env_steps():
#     env = create_env(
#         DEFAULT_NUM_DRONES,
#         active_drones = 1,
#         num_obstacles = 0,
#         world_scale = 1.0,
#     )
#
#     obs = env.reset()
#
#
#     print("\nOBS SHAPE:", obs.shape)
#
#     print("FIRST OBS:\n", obs)
#
#     """Prints full observation for each drone."""
#     np.set_printoptions(precision=3, suppress=True, linewidth=200, threshold=1000)  # Show full array
#
#     for drone_idx, state in enumerate(obs):
#         print(f"\n=== Drone {drone_idx} ===")
#         for i, val in enumerate(state):
#             print(f"{i:03d}: {val}")
#
#     def inspect_state(state):
#         print("Position:", state[:3])
#         print("Target:", state[3:6])
#         print("Front LIDAR:", state[6])
#         print("Left LIDAR:", state[7])
#         print("Right LIDAR:", state[8])
#         print("Full LIDAR slice 6–50:", state[6:50])
#
#     inspect_state(obs[0])
#
#
#     for i in range(5):
#         random_action = [env.action_space.sample()]
#         obs,reward, done, info = env.step(random_action)
#
#         print(f"\n--- step {i} ----")
#         print("OBS:", obs)
#         print("REWARD:", reward)
#         print("DONE:", done)
#
#     env.close()


if __name__ == "__main__":
    # debug_env_steps()
    t0 = time.time()
    train(device="auto")
    print(f"Training completed in {time.time() - t0:.2f} seconds")
