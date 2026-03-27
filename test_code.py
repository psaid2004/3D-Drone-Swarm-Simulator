from __future__ import annotations

import os, time
from pathlib import Path
import numpy as np
import torch
import warnings

import supersuit as ss
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback, EvalCallback, ProgressBarCallback, StopTrainingOnNoModelImprovement

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


# ---- CREATE ENV ----
def create_env(
        world_scale: float,
        num_drones: int,
        num_static_obstacles: int,
        num_dynamic_obstacles: int,
        max_episode_steps: int,
        lidar_enabled: bool,
        render_mode = None,
        vecnorm_training: bool = True,
):
    """
    Build PettingZoo ParallelEnv → SB3 VecEnv.
    RL sees normalised observations.
    Raw observations remain available through VecNormalize.get_original_obs().
    """

    env = DroneSwarm3D(
        world_scale = world_scale,
        num_drones_per_swarm=num_drones,
        num_static_obstacles=num_static_obstacles,
        num_dynamic_obstacles=num_dynamic_obstacles,
        max_steps=max_episode_steps,
        render_mode=render_mode,
        lidar= lidar_enabled,
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
        training=vecnorm_training
    )

    return vec


def unwrap_vec_normalize(env):
    while env is not None:
        if isinstance(env, VecNormalize):
            return env
        env = getattr(env, "venv", None)
    return None


def copy_vec_normalise_stats(src_env, dst_env, copy_reward_stats: bool = False) -> None:
    """
    Copy VecNormalize running statistics from src_env to dst_env by value (not reference).
    """
    src_vn = unwrap_vec_normalize(src_env)
    dst_vn = unwrap_vec_normalize(dst_env)

    if src_vn is None or dst_vn is None:
        return

    if getattr(src_vn, "obs_rms", None) is not None and getattr(dst_vn, "obs_rms", None) is not None:
        dst_vn.obs_rms.mean = np.array(src_vn.obs_rms.mean, copy=True)
        dst_vn.obs_rms.var = np.array(src_vn.obs_rms.var, copy=True)
        dst_vn.obs_rms.count = float(src_vn.obs_rms.count)

    if copy_reward_stats:
        if getattr(src_vn, "ret_rms", None) is not None and getattr(dst_vn, "ret_rms", None) is not None:
            dst_vn.ret_rms.mean = np.array(src_vn.ret_rms.mean, copy=True)
            dst_vn.ret_rms.var = np.array(src_vn.ret_rms.var, copy=True)
            dst_vn.ret_rms.count = float(src_vn.ret_rms.count)


def sync_eval_vecnormalise(train_env, eval_env) -> None:
    """ Eval env should use the same normalization stats as train env, but must not keep updating them. """
    copy_vec_normalise_stats(train_env, eval_env, copy_reward_stats=False)

    eval_vn = unwrap_vec_normalize(eval_env)
    if eval_vn is not None:
        eval_vn.training = False
        eval_vn.norm_reward = False


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
    """Build the full callback stack for one curriculum stage. """

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
        verbose=1,
    )

    stats_cb.reset_for_new_stage(stage_start_ts=model_num_timesteps, stage_info=stage_info)

    # stop_no_improve_cb = StopTrainingOnNoModelImprovement(
    #     max_no_improvement_evals=8,
    #     min_evals=3,
    #     verbose=1,
    # )

    # evaluate current policy on clean evaluate environment
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(Path(MODELS_DIR) / "best_model"),
        log_path=str(Path(MODELS_DIR) / "eval_logs"),
        eval_freq=max(10_000 // max(1, N_ENVS), 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        # callback_after_eval=stop_no_improve_cb,
        verbose=1,
    )

    return CallbackList([
        progress_cb,
        stats_cb,
        ckpt_cb,
        eval_cb,
    ])


# ---- EXPERT-BLENDED SAC TRAINING ----
class ExpertBlendedSAC(SAC):
    """
    Normal SAC for training.
    Expert blending takes place outside the actor in order for it to see raw observations.
    """

    def __init__(self, *args, **kwargs):
        self.expert = kwargs.pop("expert_policy", None)
        self._blend_ratio = float(np.clip(kwargs.pop("blend_ratio", 0.0), 0.0, 1.0))
        super().__init__(*args, **kwargs)

    @property
    def blend_ratio(self) -> float:
        return self._blend_ratio


    @blend_ratio.setter
    def blend_ratio(self, v: float):
        self._blend_ratio = float(np.clip(v, 0.0, 1.0))


    def unnormalise_obs(self, obs: np.ndarray) -> np.ndarray:
        vn = unwrap_vec_normalize(self.get_env())
        if vn is None or not getattr(vn, "norm_obs", False):
            return obs

        rms = getattr(vn, "obs_rms", None)
        if rms is None:
            return obs

        eps = getattr(vn, "epsilon", 1e-8)
        return obs * np.sqrt(rms.var + eps) + rms.mean


    def blend_with_expert(self, rl_action: np.ndarray, raw_obs: np.ndarray) -> np.ndarray:
        """ Blend RL action with expert action."""
        if self.expert is None or self._blend_ratio <= 0.0 or raw_obs is None:
            return rl_action

        raw_obs = np.asarray(raw_obs, dtype=np.float32)
        rl_action = np.asarray(rl_action, dtype=np.float32)

        if raw_obs.ndim == 1:
            raw_obs = raw_obs[None, :]
        if rl_action.ndim == 1:
            rl_action = rl_action[None, :]

        expert_action = np.array(
            [self.expert.action(obs) for obs in raw_obs],
            dtype=np.float32,
        )

        blended = (1.0 - self._blend_ratio) * rl_action + self._blend_ratio * expert_action
        blended = np.clip(blended, -1.0, 1.0)

        if blended.shape[0] == 1:
            return blended[0]
        return blended


    def predict(
            self,
            observation,
            state=None,
            episode_start=None,
            deterministic=False,
    ):
        """
        RL actor uses normalised observation.
        Expert uses raw observation from env-side cache.
        """
        rl_action, state = super().predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )

        obs_np = np.asarray(observation, dtype=np.float32)
        raw_obs = self.unnormalise_obs(obs_np)

        blended_action = self.blend_with_expert(rl_action, raw_obs)
        return blended_action, state


# ---- Build SAC model ----
def build_model(train_env, expert, device_):
    return ExpertBlendedSAC(
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
        policy_kwargs=dict(
            net_arch=NET_ARCH,
        ),
        expert_policy=expert,
        blend_ratio=INITIAL_BLEND,
    )


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
    ws0, nd0, ns0, ndyn0, steps0, steps_per_episode0, lidar_enabled0 = curr[0]
    print(
        f"\n=== Stage 0 | world={ws0} drones={nd0} "
        f"static obstacles={ns0} dynamic obstacles={ndyn0} steps={steps0} ==="
    )

    # --- SAC model with expert-blend policy ---
    expert = ExpertPolicy(world_scale=ws0, use_lidar=lidar_enabled0)

    # create the training environment
    train_env = create_env(
        world_scale=ws0,
        num_drones=nd0,
        num_static_obstacles=ns0,
        num_dynamic_obstacles=ndyn0,
        max_episode_steps=steps_per_episode0,
        lidar_enabled=lidar_enabled0,
        render_mode="human",
        vecnorm_training=True,
    )

    # create an evaluation environment
    eval_env = create_env(
        world_scale=ws0,
        num_drones=nd0,
        num_static_obstacles=ns0,
        num_dynamic_obstacles=ndyn0,
        max_episode_steps=steps_per_episode0,
        lidar_enabled=lidar_enabled0,
        render_mode=None,
        vecnorm_training=False,
    )

    sync_eval_vecnormalise(train_env, eval_env)

    # --- create SAC model ---
    model = ExpertBlendedSAC(
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
        policy_kwargs=dict(
            net_arch=NET_ARCH,
        ),
        expert_policy=expert,
        blend_ratio=INITIAL_BLEND,
    )# TODO WE NEED TO DO SOME SORT OF HYPERPARAMETER TRAINING HERE TO FIND THE OPTIMAL VALUES

    # --- train stage 0 first ---

    callbacks = make_stage_callbacks(
        stage_idx=0,
        world_scale=ws0,
        num_drones=nd0,
        num_static_obs=ns0,
        num_dynamic_obs=ndyn0,
        stage_steps=steps0,
        episode_horizon = steps_per_episode0,
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
    for idx, (world_scale, num_drones, num_static_obs, num_dynamic_obs, stage_steps, steps_per_episode, lidar_enabled) in enumerate(curr[1:], start=1):
        stage_steps = int(stage_steps)

        print(
            f"\n=== Stage {idx} | world={world_scale} "
            f"drones={num_drones} static={num_static_obs} "
            f"dynamic={num_dynamic_obs} steps={stage_steps} ==="
        )

        old_train_env = model.get_env()
        old_eval_env = eval_env
        old_model = model

        # update expert to match new curriculum stage
        expert.world_scale = world_scale
        expert.use_lidar = lidar_enabled

        new_train_env = create_env(
            world_scale=world_scale,
            num_drones=num_drones,
            num_static_obstacles=num_static_obs,
            num_dynamic_obstacles=num_dynamic_obs,
            max_episode_steps=steps_per_episode,
            lidar_enabled=lidar_enabled,
            render_mode=None,
            vecnorm_training=True,
        )

        new_eval_env = create_env(
            world_scale=world_scale,
            num_drones=num_drones,
            num_static_obstacles=num_static_obs,
            num_dynamic_obstacles=num_dynamic_obs,
            max_episode_steps=steps_per_episode,
            lidar_enabled=lidar_enabled,
            render_mode=None,
            vecnorm_training=False,
        )
        # Carry normalization stats forward.
        copy_vec_normalise_stats(old_train_env, new_train_env, copy_reward_stats=False)

        # eval env should mirror train stats
        sync_eval_vecnormalise(new_train_env, new_eval_env)

        # rebuild model instead of set_env
        model = build_model(
            train_env=new_train_env,
            expert=expert,
            device_=device_,
        )

        # copy weights
        model.policy.load_state_dict(old_model.policy.state_dict())
        model.actor.load_state_dict(old_model.actor.state_dict())
        model.critic.load_state_dict(old_model.critic.state_dict())
        model.critic_target.load_state_dict(old_model.critic_target.state_dict())

        # carry timesteps
        model.num_timesteps = old_model.num_timesteps
        model._total_timesteps = old_model._total_timesteps
        model._episode_num = old_model._episode_num

        eval_env = new_eval_env
        train_env = new_train_env

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
