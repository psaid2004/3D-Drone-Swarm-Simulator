from __future__ import annotations

import os, time
from pathlib import Path
import numpy as np

import torch
import warnings

import supersuit as ss
from stable_baselines3 import SAC

#UPDATED CALLBACKS
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    ProgressBarCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.sac.policies import SACPolicy

from config.curriculum import Curriculum
from config.defaults import *
from config.sac_defaults import *

from train.callbacks.custom_stats_cb import StepStatsCallback, CurriculumStageInfo
from wrappers.env_3d import DroneSwarm3D
from wrappers.env_reset_fix import SB3VecResetTupleFix
from train.expert_blender import ExpertPolicy

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


# ---- 3-D EXPERT BLEND POLICY ----
class ExpertBlendPolicy(SACPolicy):
    """
    SACPolicy whose action selection blends RL + expert.
    Gradient updates use standard SAC on the replay buffer.
    """

    def __init__(self, *args, expert_policy=None, blend_ratio=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert = expert_policy
        self._blend_ratio = float(blend_ratio)


    @property
    def blend_ratio(self) -> float:
        return self._blend_ratio


    @blend_ratio.setter
    def blend_ratio(self, v: float):
        self._blend_ratio = float(np.clip(v, 0.0, 1.0))

    #CHANGED
    def _predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # RL action from SAC actor
        rl_action = self.actor(obs, deterministic=deterministic)

        # no expert / no blending -> pure RL
        if self.expert is None or self._blend_ratio <= 0.0:
            return rl_action

        # expert action for each item in batch
        obs_np = obs.detach().cpu().numpy()
        expert_np = np.array(
            [self.expert.action(o) for o in obs_np],
            dtype=np.float32,
        )
        expert_t = torch.as_tensor(expert_np, device=obs.device)

        blended = self._blend_ratio * expert_t + (1.0 - self._blend_ratio) * rl_action
        return torch.clamp(blended, -1.0, 1.0)

#ADDED CLASS
class BlendScheduleCallback(BaseCallback):
    """
    Linearly decay expert blending from INITIAL_BLEND to FINAL_BLEND
    over BLEND_DECAY_FRAC of the total planned curriculum timesteps.
    """

    def __init__(
        self,
        total_planned_timesteps: int,
        initial_blend: float = INITIAL_BLEND,
        final_blend: float = FINAL_BLEND,
        decay_frac: float = BLEND_DECAY_FRAC,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.total_planned_timesteps = int(total_planned_timesteps)
        self.initial_blend = float(initial_blend)
        self.final_blend = float(final_blend)
        self.decay_frac = float(decay_frac)

    def _current_blend(self) -> float:
        decay_steps = max(1, int(self.total_planned_timesteps * self.decay_frac))
        t = min(int(self.model.num_timesteps), decay_steps)
        alpha = t / decay_steps
        blend = self.initial_blend + alpha * (self.final_blend - self.initial_blend)
        return float(np.clip(blend, 0.0, 1.0))

    def _set_blend(self) -> None:
        if hasattr(self.model, "policy") and hasattr(self.model.policy, "blend_ratio"):
            blend = self._current_blend()
            self.model.policy.blend_ratio = blend
            self.logger.record("expert_blend/blend_ratio", blend)

    def _on_training_start(self) -> None:
        self._set_blend()

    def _on_step(self) -> bool:
        self._set_blend()
        return True

# ---- CREATE ENV ---- updated
def create_env(
    world_scale,
    num_drones,
    num_static_obstacles,
    num_dynamic_obstacles,
    max_episode_steps,
    lidar_enabled,
    render_mode=None,
):
    """Build PettingZoo ParallelEnv → SB3 VecEnv."""

    env = DroneSwarm3D(
        world_scale=world_scale,
        num_drones_per_swarm=num_drones,
        num_static_obstacles=num_static_obstacles,
        num_dynamic_obstacles=num_dynamic_obstacles,
        max_steps=max_episode_steps,
        lidar=lidar_enabled,
        render_mode=render_mode,
    )

    vec = ss.pettingzoo_env_to_vec_env_v1(env)
    vec = ss.concat_vec_envs_v1(vec, N_ENVS, base_class="stable_baselines3")

    vec = SB3VecResetTupleFix(vec)
    vec = VecMonitor(vec)

    # IMPORTANT:
    # ExpertPolicy expects raw semantic observations, so norm_obs must stay False.
    vec = VecNormalize(
        vec,
        gamma=GAMMA,
        norm_reward=False,
        norm_obs=False,
        clip_obs=5.0,
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
    episode_horizon: int,
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
        episode_horizon = int(episode_horizon),
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
        # print_every=5000,
        enable_stage_early_stop=True,
        reach_threshold=0.85,
        max_collision_rate=0.05,
        patience=10,
        min_stage_timesteps=10000,
        verbose=0,
    )

    stats_cb.reset_for_new_stage(stage_start_ts=model_num_timesteps, stage_info=stage_info)

    #ADDED
    blend_cb = BlendScheduleCallback(
        total_planned_timesteps=total_planned_timesteps,
        initial_blend=INITIAL_BLEND,
        final_blend=FINAL_BLEND,
        decay_frac=BLEND_DECAY_FRAC,
        verbose=0,
    )

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

    #UPDATED
    return CallbackList([
        blend_cb,
        progress_cb,
        stats_cb,
        ckpt_cb,
        # eval_cb,
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
    print(torch.cuda.is_available())

    curr = Curriculum().stages
    total_planned_timesteps = int(sum(int(s[4]) for s in curr)) #sum of all curriculum stage timesteps

    # --- build stage-0 env ---

    #UPDATED
    ws0, nd0, ns0, ndyn0, steps0, steps_per_episode, lidar0 = curr[0]
    print(
        f"\n=== Stage 0 | world={ws0} drones={nd0} "
        f"static obstacles={ns0} dynamic obstacles={ndyn0} steps={steps0} ==="
    )

    # --- SAC model with expert-blend policy ---
    # TODO ADD THIS BELOW
    # TODO GET LIDAR FROM CURRICULUM
    # TODO THERE IS SOME ISSUE FOR THE NORM_OBS = TRUE CASE

    #CHANGED
    expert = ExpertPolicy(world_scale=ws0, use_lidar=lidar0)

    train_env = create_env(
        world_scale=ws0,
        num_drones=nd0,
        num_static_obstacles=ns0,
        num_dynamic_obstacles=ndyn0,
        max_episode_steps=steps_per_episode,
        lidar_enabled=lidar0,
        render_mode=None,
    )

    eval_env = create_env(
        world_scale=ws0,
        num_drones=nd0,
        num_static_obstacles=ns0,
        num_dynamic_obstacles=ndyn0,
        max_episode_steps=steps_per_episode,
        lidar_enabled=lidar0,
        render_mode=None,
    )

    # --- create SAC model ---
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
        tensorboard_log=f"{MODELS_DIR}/tb_logs",
        verbose=1,
        device=device_,
        policy_kwargs=dict(
            net_arch=NET_ARCH,
            expert_policy=expert,
            blend_ratio=INITIAL_BLEND,
        ),
    )# TODO WE NEED TO DO SOME SORT OF HYPERPARAMETER TRAINING HERE TO FIND THE OPTIMAL VALUES

    # --- train stage 0 first ---

    callbacks = make_stage_callbacks(
        stage_idx=0,
        world_scale=ws0,
        num_drones=nd0,
        num_static_obs=ns0,
        num_dynamic_obs=ndyn0,
        stage_steps=steps0,
        episode_horizon = steps_per_episode,
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
    for idx, (
            world_scale,
            num_drones,
            num_static_obs,
            num_dynamic_obs,
            stage_steps,
            steps_per_episode,
            lidar_enabled,
    ) in enumerate(curr[1:], start=1):
        stage_steps = int(stage_steps)

        print(
            f"\n=== Stage {idx} | world={world_scale} "
            f"drones={num_drones} static={num_static_obs} "
            f"dynamic={num_dynamic_obs} steps={stage_steps} ==="
        )

        # swap envs
        old_train_env = model.get_env()
        old_eval_env = eval_env

        #CHANGE
        new_train_env = create_env(
            world_scale=world_scale,
            num_drones=num_drones,
            num_static_obstacles=num_static_obs,
            num_dynamic_obstacles=num_dynamic_obs,
            max_episode_steps=steps_per_episode,
            lidar_enabled=lidar_enabled,
        )

        new_eval_env = create_env(
            world_scale=world_scale,
            num_drones=num_drones,
            num_static_obstacles=num_static_obs,
            num_dynamic_obstacles=num_dynamic_obs,
            max_episode_steps=steps_per_episode,
            lidar_enabled=lidar_enabled,
        )

        # Carry normalization stats forward.
        copy_vec_normalize_stats(old_train_env, new_train_env)
        copy_vec_normalize_stats(old_train_env, new_eval_env)

        model.set_env(new_train_env)
        eval_env = new_eval_env

        #CHANGE
        # update expert to match the current curriculum stage
        if hasattr(model.policy, "expert") and model.policy.expert is not None:
            model.policy.expert.world_scale = float(world_scale)
            model.policy.expert.use_lidar = bool(lidar_enabled)

        callbacks = make_stage_callbacks(
            stage_idx=idx,
            world_scale=world_scale,
            num_drones=num_drones,
            num_static_obs=num_static_obs,
            num_dynamic_obs=num_dynamic_obs,
            stage_steps=stage_steps,
            episode_horizon = steps_per_episode,
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
