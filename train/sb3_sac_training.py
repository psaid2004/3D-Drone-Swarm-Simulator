"""
SB3 SAC training (3D PyBullet drones) with expert blending.

All required defaults and curriculum stages are imported from the config file.

Run:
  python -m train.sb3_sac_training
"""

from __future__ import annotations

import os, time
import numpy as np
import torch

import supersuit as ss
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback
)
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.sac.policies import SACPolicy

from config.curriculum import Curriculum
from config.defaults import (DEFAULT_NUM_DRONES, MODELS_DIR, LIDAR_NUM_LAYERS,
                             LIDAR_NUM_RAYS, LIDAR_RANGE, DEFAULT_LIDAR, eps)
from train.callbacks.stats_cb import StepStatsCallback, CurriculumStageInfo, BlendDecayCallback, EarlyStoppingCallback
from wrappers.pz_parallel_3d import env as make_env
from wrappers.sb3_reset_fix import SB3VecResetTupleFix

# ---- PATHS ----
MODEL_DIR = MODELS_DIR

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

# ---- EXPERT CONFIG ----
INITIAL_BLEND   = 0.8        # expert weight at start
FINAL_BLEND     = 0.0        # pure RL at end
BLEND_DECAY_FRAC = 0.5       # fraction of total timesteps to anneal over

# ---- FILESYSTEM HELPERS ----
def get_next_model_path(base=f"{MODEL_DIR}/sac_", drones=1, obs=0, ext=".zip"):
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


# ---- 3-D EXPERT POLICY ----
#
# Observation layout (from DroneSwarm3D._build_obs_dict):
#   [0:3]   pos
#   [3:6]   vel
#   [6:9]   to_target        (target − pos)
#   [9:12]  to_nearest_obs   (nearest obstacle − pos)
#   [12:15] to_centroid      (swarm centre − pos)
#   [15:18] to_nearest_teammate
#   [18: ]  lidar hit fractions  (LIDAR_NUM_LAYERS * LIDAR_NUM_RAYS)
#
# Action space: [dir_x, dir_y, dir_z, speed]  all in [-1, 1]

class ExpertPolicy:
    """Reactive 3-D expert: steer toward target, dodge obstacles via LiDAR."""
    # TODO IF OBSTACLES ARE NOT BEING AVOIDED, WE NEED TO DECREASE THE SPEED HERE AND INCREASE THE AVOID_STRENGTH
    def __init__(self, avoid_strength: float = 0.6, speed_default: float = 0.6):
        self.avoid_strength = avoid_strength
        self.speed_default = speed_default

    def expert_action(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)

        # --- target direction ---
        to_target = state[6:9].copy()
        # compute Euclidean distance to target
        dist = np.linalg.norm(to_target) + eps
        # normalise into unit vector pointing at target
        dir_target = to_target / dist

        # --- obstacle vector (points FROM drone to nearest obstacle) ---
        to_obs = state[9:12]
        # compute Euclidean distance to nearest obstacle
        obs_dist = np.linalg.norm(to_obs) + eps

        # --- lidar-based avoidance (if lidar present) ---
        lidar_start = 18
        n_lidar = LIDAR_NUM_LAYERS * LIDAR_NUM_RAYS
        # initialise avoidance direction vector
        avoid = np.zeros(3, dtype=np.float32)

        # check whether observation contains LiDAR
        if state.shape[0] >= lidar_start + n_lidar and n_lidar > 0:
            lidar = state[lidar_start:lidar_start + n_lidar]
            # reshape LiDAR to 2D
            lidar = lidar.reshape(LIDAR_NUM_LAYERS, LIDAR_NUM_RAYS)

            # use middle layer for horizontal avoidance
            mid = lidar[LIDAR_NUM_LAYERS // 2]

            # angles for each ray (full 360° by default)
            angles = np.linspace(0, 2 * np.pi, LIDAR_NUM_RAYS, endpoint=False)

            # weight: closer hits (small fraction) → stronger repulsion
            proximity = np.clip(1.0 - mid, 0, 1)  # 1 = contact, 0 = clear

            # repulsion vector: push away from close hits
            # weight the ray angles by the closeness, sum it to get net "pull" and then negative gives net "push"
            # (rx, ry) points away from where nearby obstacles are concentrated
            rx = -np.sum(proximity * np.cos(angles))
            ry = -np.sum(proximity * np.sin(angles))

            # vertical avoidance from top / bottom layers
            top_prox = np.clip(1.0 - lidar[0].mean(), 0, 1)
            bot_prox = np.clip(1.0 - lidar[-1].mean(), 0, 1)
            rz = bot_prox - top_prox  # push up if bottom blocked, down if top

            # create the avoidance array for both horizontal and vertical repulsion
            avoid = np.array([rx, ry, rz], dtype=np.float32)
            # compute Euclidean distance
            an = np.linalg.norm(avoid) + 1e-8
            # if not tiny (noise), then normalise it into a unit vector
            if an > 0.01:
                avoid = avoid / an

        # --- also add repulsion from nearest-obstacle vector ---
        if obs_dist < LIDAR_RANGE:
            # determine unit vector pointing away from obstacle
            obs_repel = -to_obs / obs_dist
            # compute closeness of obstacle
            weight = max(0, 1.0 - obs_dist / LIDAR_RANGE) # 1 = contact, 0 = clear
            # add to the current avoid vector from LiDAR
            avoid = avoid + weight * obs_repel
            # compute Euclidean and normalise
            an = np.linalg.norm(avoid) + 1e-8
            if an > 1e-3:
                avoid /= an

        # --- blend target-seeking with avoidance ---
        action_dir = dir_target + self.avoid_strength * avoid
        n = np.linalg.norm(action_dir) + 1e-8
        action_dir = action_dir / n

        # --- speed: slow down near obstacles, brake near target ---
        speed = self.speed_default
        if obs_dist < 2.0:
            speed *= 0.2
        if dist < 1.5:
            speed *= 0.3

        # map speed from [0,1] into the action-space convention [-1, 1]
        speed_action = float(np.clip(speed * 2.0 - 1.0, -1.0, 1.0))

        # return the [dx, dy, dz, speed] expert action
        return np.array([
            float(np.clip(action_dir[0], -1, 1)),
            float(np.clip(action_dir[1], -1, 1)),
            float(np.clip(action_dir[2], -1, 1)),
            speed_action,
        ], dtype=np.float32)


# ---- BLEND POLICY ----
class ExpertBlendPolicy(SACPolicy):
    """
    SACPolicy whose action selection blends RL + expert.
    Gradient updates use standard SAC on the replay buffer.
    """

    def __init__(self, *args, expert_policy=None, blend_ratio=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert = expert_policy or ExpertPolicy()
        self._blend_ratio = float(blend_ratio)

    @property
    def blend_ratio(self) -> float:
        return self._blend_ratio

    @blend_ratio.setter
    def blend_ratio(self, v: float):
        self._blend_ratio = float(np.clip(v, 0.0, 1.0))

    def _predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # --- RL action from the SAC actor ---
        # retrieve the SAC action
        rl_action = self.actor(obs, deterministic=deterministic)

        # if blend ratio is zero or less, only follow SAC action
        if self._blend_ratio <= 0.0:
            return rl_action

        # --- expert action (per observation in batch) ---
        obs_np = obs.detach().cpu().numpy()
        # determine the expert action for each observation in the batched sample
        expert_np = np.array(
            [self.expert.expert_action(o) for o in obs_np],
            dtype=np.float32,
        )
        # create tensor
        expert_t = torch.as_tensor(expert_np, device=obs.device)

        # blend the expert action with the SAC action
        blended = self._blend_ratio * expert_t + (1.0 - self._blend_ratio) * rl_action

        return torch.clamp(blended, -1.0, 1.0)


# ---- CREATE ENV ----
def create_env(num_drones: int, active_drones: int, num_obstacles: int, world_scale: float):
    """Build PettingZoo ParallelEnv → SB3 VecEnv."""

    raw = make_env(
        num_drones=num_drones,
        active_drones=active_drones,
        num_obstacles=num_obstacles,
        max_steps=1000,
        lidar=DEFAULT_LIDAR,
        render_mode=None,
        world_scale=world_scale,
    )

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

    # --- SAC model with expert-blend policy ---
    expert = ExpertPolicy()

    model = SAC(
        ExpertBlendPolicy,
        train_env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        gamma=GAMMA,
        ent_coef=ENTROPY_COEF,
        policy_kwargs=dict(
            net_arch=NET_ARCH,
            expert_policy=expert,
            blend_ratio=INITIAL_BLEND,
        ),
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

        # transfer normalisation stats
        if isinstance(old_env, VecNormalize) and isinstance(new_env, VecNormalize):
            new_env.obs_rms = old_env.obs_rms
            new_env.ret_rms = old_env.ret_rms
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
            verbose=1,
        )

        blend_cb = BlendDecayCallback(
            total_timesteps=total_planned,
            initial=INITIAL_BLEND,
            final=FINAL_BLEND,
            decay_frac=BLEND_DECAY_FRAC,
            verbose=1,
        )

        # TODO TEST THIS STILL
        early_cb = EarlyStoppingCallback(
            reach_threshold=0.85,
            max_collision_rate=0.05,
            patience=10,
            min_timesteps=10_000,
            verbose=1,
        )
        early_cb.reset(stage_start_ts=model.num_timesteps)

        callbacks = CallbackList([stats_cb, ckpt_cb, blend_cb, early_cb])

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

    train_env.close()
    print("Training complete.")


if __name__ == "__main__":
    t0 = time.time()
    train(device="auto")
    print(f"Finished in {time.time() - t0:.1f}s")
