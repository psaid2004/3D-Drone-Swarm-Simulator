from __future__ import annotations

import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from config.curriculum import Curriculum
from config.defaults import MAX_DRONES_PER_SWARM
from config.sac_defaults import (
    LEARNING_RATE,
    BUFFER_SIZE,
    BATCH_SIZE,
    TRAIN_FREQ,
    GAMMA,
    ENTROPY_COEF,
    NET_ARCH,
    INITIAL_BLEND,
    N_ENVS,
)

from train.sac_training_with_expert import (
    create_env,
    ExpertBlendPolicy,
    ActiveSlotReplayBuffer,
    unwrap_vec_normalize,
)
from train.expert_blender import ExpertPolicy


def is_zero_obs(x: np.ndarray, atol: float = 1e-8) -> bool:
    return bool(np.all(np.abs(x) <= atol))


def frozen_action() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)


class ReplayBufferInspectorCallback(BaseCallback):
    """
    Prints:
    - normalized obs actually seen by the policy
    - denormalized/raw obs
    - info flags
    - expert / RL / final blended actions
    - latest replay-buffer insertions
    - active_mask stored in replay buffer
    - sampled replay-buffer minibatch stats
    """

    def __init__(
        self,
        stage_idx: int,
        active_drones: int,
        max_drones: int,
        num_parallel_envs: int,
        print_every_steps: int = 1,
        max_print_steps: int = 50,
        sample_batch_size: int = 32,
        max_slots_to_show: int = 12,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.stage_idx = int(stage_idx)
        self.active_drones = int(active_drones)
        self.max_drones = int(max_drones)
        self.num_parallel_envs = int(num_parallel_envs)
        self.print_every_steps = int(print_every_steps)
        self.max_print_steps = int(max_print_steps)
        self.sample_batch_size = int(sample_batch_size)
        self.max_slots_to_show = int(max_slots_to_show)
        self._step_counter = 0

    def _print_header(self) -> None:
        total_slots = self.training_env.num_envs
        print("\n" + "=" * 110)
        print(
            f"[DEBUG] stage={self.stage_idx} | "
            f"active_drones={self.active_drones} | "
            f"max_drones={self.max_drones} | "
            f"N_ENVS(config)={self.num_parallel_envs} | "
            f"SB3 vec slots seen by model={total_slots}"
        )
        print("=" * 110)

    def _on_training_start(self) -> None:
        self._print_header()

    def _on_step(self) -> bool:
        self._step_counter += 1

        if self._step_counter > self.max_print_steps:
            return False

        if self._step_counter % self.print_every_steps != 0:
            return True

        infos = self.locals.get("infos", None)
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        new_obs = self.locals.get("new_obs", None)

        if infos is None or new_obs is None:
            print(f"[DEBUG step {self._step_counter}] infos/new_obs unavailable")
            return True

        vn = unwrap_vec_normalize(self.training_env)

        # what the model sees
        policy_obs = new_obs.copy()

        # approximate raw obs for debugging
        if vn is not None and getattr(vn, "norm_obs", False):
            try:
                raw_obs = vn.unnormalize_obs(new_obs.copy())
            except Exception:
                raw_obs = new_obs.copy()
        else:
            raw_obs = new_obs.copy()

        print(f"\n--- DEBUG STEP {self._step_counter} ---")
        print(f"len(infos)={len(infos)}")
        print(f"policy_obs.shape={np.shape(policy_obs)}")
        print(f"raw_obs.shape={np.shape(raw_obs)}")

        max_slots_to_show = min(len(infos), self.max_slots_to_show)

        # ------------------------------------------------------------------
        # 1) Inspect rollout observations and infos
        # ------------------------------------------------------------------
        for flat_idx in range(max_slots_to_show):
            info = infos[flat_idx]
            obs_norm = policy_obs[flat_idx]
            obs_raw = raw_obs[flat_idx]

            active = bool(info.get("active", False))
            collided = bool(info.get("collided", False))
            reached = bool(info.get("reached", False))
            cmd_speed = float(info.get("cmd_speed", 0.0))
            actual_speed = float(info.get("actual_speed", 0.0))
            dist_to_target = float(info.get("dist_to_target", 0.0))

            env_i = flat_idx // self.max_drones
            drone_i = flat_idx % self.max_drones

            print(
                f"[slot {flat_idx:02d}] env={env_i} drone={drone_i} "
                f"active={active} reached={reached} collided={collided} "
                f"zero_policy_obs={is_zero_obs(obs_norm)} zero_raw={is_zero_obs(obs_raw)} "
                f"reward={float(rewards[flat_idx]) if rewards is not None else 0.0:.3f} "
                f"done={bool(dones[flat_idx]) if dones is not None else False} "
                f"dist={dist_to_target:.3f} cmd_speed={cmd_speed:.3f} actual_speed={actual_speed:.3f}"
            )
            print(f"    raw_obs[:12]    = {np.array2string(obs_raw[:12], precision=3, suppress_small=True)}")
            print(f"    policy_obs[:12] = {np.array2string(obs_norm[:12], precision=3, suppress_small=True)}")

        # ------------------------------------------------------------------
        # 2) Print expert / RL / final action comparison
        # ------------------------------------------------------------------
        try:
            obs_tensor = torch.as_tensor(policy_obs, device=self.model.device, dtype=torch.float32)

            with torch.no_grad():
                rl_action_t = self.model.policy.actor(obs_tensor, deterministic=True)
                final_action_t = self.model.policy._predict(obs_tensor, deterministic=True)

            rl_action = rl_action_t.detach().cpu().numpy()
            final_action = final_action_t.detach().cpu().numpy()

            raw_for_expert = self.model.policy.denormalise_observation(policy_obs.copy())

            expert_actions = []
            for flat_idx in range(len(infos)):
                if is_zero_obs(policy_obs[flat_idx]):
                    expert_a = frozen_action()
                else:
                    expert_a = self.model.policy.expert.action(raw_for_expert[flat_idx])
                expert_actions.append(expert_a)
            expert_actions = np.asarray(expert_actions, dtype=np.float32)

            print(f"\n[Action comparison] blend_ratio={self.model.policy.blend_ratio:.3f}")
            for flat_idx in range(max_slots_to_show):
                info = infos[flat_idx]
                active = bool(info.get("active", False))
                env_i = flat_idx // self.max_drones
                drone_i = flat_idx % self.max_drones

                print(
                    f"[slot {flat_idx:02d}] env={env_i} drone={drone_i} active={active}\n"
                    f"    expert = {np.array2string(expert_actions[flat_idx], precision=3, suppress_small=True)}\n"
                    f"    rl     = {np.array2string(rl_action[flat_idx], precision=3, suppress_small=True)}\n"
                    f"    final  = {np.array2string(final_action[flat_idx], precision=3, suppress_small=True)}"
                )
        except Exception as e:
            print(f"[DEBUG] action comparison failed: {e}")

        # ------------------------------------------------------------------
        # 3) Inspect replay buffer
        # ------------------------------------------------------------------
        rb = self.model.replay_buffer
        if rb is None:
            print("[DEBUG] replay buffer is None")
            return True

        print("\n[Replay buffer summary]")
        print(f"buffer_size={rb.buffer_size}")
        print(f"pos={rb.pos}")
        print(f"full={rb.full}")

        current_size = rb.buffer_size if rb.full else rb.pos
        print(f"current_size={current_size}")

        if current_size > 0:
            last_idx = (rb.pos - 1) % rb.buffer_size

            obs_last = rb.observations[last_idx]
            acts_last = rb.actions[last_idx]
            rews_last = rb.rewards[last_idx]
            dones_last = rb.dones[last_idx]

            next_obs_last = None
            if hasattr(rb, "next_observations") and rb.next_observations is not None:
                next_obs_last = rb.next_observations[last_idx]

            print(f"last stored slot index={last_idx}")
            print(f"obs_last.shape={obs_last.shape}")
            print(f"actions_last.shape={acts_last.shape}")

            if hasattr(rb, "active_mask"):
                active_last = rb.active_mask[last_idx]
                print(f"active_mask last insert = {active_last.astype(np.int32)}")
                print(f"active slots in last insert = {int(np.sum(active_last))}/{len(active_last)}")

            zero_obs_count = 0
            frozen_action_count = 0
            for slot in range(obs_last.shape[0]):
                if is_zero_obs(obs_last[slot]):
                    zero_obs_count += 1
                if np.allclose(acts_last[slot], frozen_action(), atol=1e-6):
                    frozen_action_count += 1

            print(f"zero-observation rows in latest replay insert = {zero_obs_count}/{obs_last.shape[0]}")
            print(f"frozen-action rows in latest replay insert    = {frozen_action_count}/{acts_last.shape[0]}")

            for slot in range(min(obs_last.shape[0], self.max_slots_to_show)):
                env_i = slot // self.max_drones
                drone_i = slot % self.max_drones
                print(
                    f"[rb last slot {slot:02d}] env={env_i} drone={drone_i} "
                    f"zero_obs={is_zero_obs(obs_last[slot])} "
                    f"rew={float(rews_last[slot]):.3f} done={bool(dones_last[slot])}"
                )
                print(f"    obs[:10] = {np.array2string(obs_last[slot][:10], precision=3, suppress_small=True)}")
                if next_obs_last is not None:
                    print(f"    next[:10]= {np.array2string(next_obs_last[slot][:10], precision=3, suppress_small=True)}")
                print(f"    act      = {np.array2string(acts_last[slot], precision=3, suppress_small=True)}")

        # ------------------------------------------------------------------
        # 4) Inspect sampled minibatch
        # ------------------------------------------------------------------
        if current_size >= max(2, self.sample_batch_size):
            try:
                batch = rb.sample(self.sample_batch_size)
                obs_sample = batch.observations.detach().cpu().numpy()
                act_sample = batch.actions.detach().cpu().numpy()
                rew_sample = batch.rewards.detach().cpu().numpy()

                zero_rows = sum(is_zero_obs(x) for x in obs_sample)
                frozen_rows = sum(np.allclose(a, frozen_action(), atol=1e-6) for a in act_sample)

                print("\n[Sampled minibatch]")
                print(f"sample_batch_size={self.sample_batch_size}")
                print(f"zero-observation samples = {zero_rows}/{len(obs_sample)}")
                print(f"frozen-action samples    = {frozen_rows}/{len(act_sample)}")
                print(
                    f"reward stats: mean={float(np.mean(rew_sample)):.4f}, "
                    f"min={float(np.min(rew_sample)):.4f}, max={float(np.max(rew_sample)):.4f}"
                )

                for i in range(min(len(obs_sample), 8)):
                    print(
                        f"[sample {i:02d}] zero_obs={is_zero_obs(obs_sample[i])} "
                        f"rew={float(rew_sample[i]):.3f}"
                    )
                    print(f"    obs[:10] = {np.array2string(obs_sample[i][:10], precision=3, suppress_small=True)}")
                    print(f"    act      = {np.array2string(act_sample[i], precision=3, suppress_small=True)}")
            except Exception as e:
                print(f"[DEBUG] replay buffer sample failed: {e}")

        return True


def run_debug_curriculum(
    n_stages: int = 5,
    episodes_per_stage: int = 10,
    max_debug_steps_per_stage: int = 100,
):
    curr = Curriculum().stages[:n_stages]

    for stage_idx, stage in enumerate(curr):
        (
            world_scale,
            num_drones,
            num_static_obs,
            num_dynamic_obs,
            stage_train_steps,
            episode_horizon,
            lidar_enabled,
        ) = stage

        print("\n" + "#" * 120)
        print(
            f"STAGE {stage_idx} | "
            f"world_scale={world_scale}, active_drones={num_drones}, "
            f"static={num_static_obs}, dynamic={num_dynamic_obs}, "
            f"episode_horizon={episode_horizon}, lidar={lidar_enabled}"
        )
        print("#" * 120)

        expert = ExpertPolicy(world_scale=world_scale, use_lidar=lidar_enabled)

        train_env = create_env(
            world_scale=world_scale,
            num_drones=num_drones,
            num_static_obstacles=num_static_obs,
            num_dynamic_obstacles=num_dynamic_obs,
            max_episode_steps=episode_horizon,
            lidar_enabled=lidar_enabled,
            render_mode=None,
        )

        model = SAC(
            ExpertBlendPolicy,
            train_env,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            learning_starts=0,
            batch_size=BATCH_SIZE,
            train_freq=TRAIN_FREQ,
            gradient_steps=1,
            gamma=GAMMA,
            ent_coef=ENTROPY_COEF,
            verbose=0,
            replay_buffer_class=ActiveSlotReplayBuffer,
            policy_kwargs=dict(
                net_arch=NET_ARCH,
                expert_policy=expert,
                blend_ratio=INITIAL_BLEND,
            ),
            device="cpu",
        )

        vn = unwrap_vec_normalize(train_env)
        if vn is not None:
            model.policy.set_vec_normalise(vn)

        total_slots = train_env.num_envs
        steps_to_run = min(
            max_debug_steps_per_stage,
            episodes_per_stage * episode_horizon,
        )

        print(f"SB3 sees train_env.num_envs = {total_slots}")
        print(
            "Expected flattened slots = "
            f"MAX_DRONES_PER_SWARM({MAX_DRONES_PER_SWARM}) * N_ENVS({N_ENVS}) = {MAX_DRONES_PER_SWARM * N_ENVS}"
        )

        debug_cb = ReplayBufferInspectorCallback(
            stage_idx=stage_idx,
            active_drones=num_drones,
            max_drones=MAX_DRONES_PER_SWARM,
            num_parallel_envs=N_ENVS,
            print_every_steps=1,
            max_print_steps=steps_to_run,
            sample_batch_size=min(32, BATCH_SIZE),
            max_slots_to_show=12,
            verbose=0,
        )

        model.learn(
            total_timesteps=steps_to_run,
            reset_num_timesteps=True,
            callback=debug_cb,
            progress_bar=False,
        )

        train_env.close()

    print("\n[DONE] debug curriculum run finished.")


if __name__ == "__main__":
    run_debug_curriculum(
        n_stages=5,
        episodes_per_stage=10,
        max_debug_steps_per_stage=100,
    )