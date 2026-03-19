"""
GUI test for DroneSwarm3D with reward function.

Run:
    python -m train.test_env_gui
"""

from __future__ import annotations

import time
import numpy as np
from wrappers.env_3d import DroneSwarm3D


def safe_unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return np.zeros_like(v) if norm < 1e-6 else v / norm


class EnvGUITester:
    """Runs one DroneSwarm3D instance with GUI and prints reward diagnostics."""

    def __init__(
        self,
        world_scale: float = 5.0,
        num_drones: int = 1,
        num_static_obstacles: int = 0,
        num_dynamic_obstacles: int = 0,
        max_steps: int = 500,
        n_episodes: int = 3,
    ):
        self.n_episodes = n_episodes
        self.max_steps = max_steps

        self.env = DroneSwarm3D(
            world_scale=world_scale,
            num_drones_per_swarm=num_drones,
            num_static_obstacles=num_static_obstacles,
            num_dynamic_obstacles=num_dynamic_obstacles,
            max_steps=max_steps,
            num_swarms=1,
            lidar=False,
            render_mode="human",
        )

        print(
            f"[EnvGUITester] world={world_scale} | drones={num_drones} | "
            f"static_obs={num_static_obstacles} | dynamic_obs={num_dynamic_obstacles} | "
            f"max_steps={max_steps}"
        )

    # ------------------------------------------------------------------
    # Action policy: move toward target, weighted by centroid cohesion
    # ------------------------------------------------------------------
    def _get_action(self, obs: np.ndarray) -> np.ndarray:
        to_target = obs[6:9]
        to_centroid = obs[12:15]

        dir_vec = safe_unit(0.8 * safe_unit(to_target) + 0.2 * safe_unit(to_centroid))
        speed = float(np.clip(np.linalg.norm(to_target) / 5.0, 0.05, 1.0))
        u3 = speed * 2.0 - 1.0  # map [0,1] → [-1,1]

        return np.array([dir_vec[0], dir_vec[1], dir_vec[2], u3], dtype=np.float32)

    # ------------------------------------------------------------------
    # Single episode
    # ------------------------------------------------------------------
    def _run_episode(self, ep: int) -> dict:
        obs_dict, _ = self.env.reset()

        print(f"\n{'='*50}")
        print(f"  EPISODE {ep + 1}")
        print(f"{'='*50}")

        episode_rewards = {d: 0.0 for d in obs_dict}
        step_count = 0

        for step in range(self.max_steps):
            actions = {d: self._get_action(obs) for d, obs in obs_dict.items()}
            obs_dict, reward_dict, term_dict, trunc_dict, info_dict = self.env.step(actions)

            for d, r in reward_dict.items():
                episode_rewards[d] += r

            # Print diagnostics every 50 steps
            if step % 50 == 0:
                self._print_step(step, obs_dict, reward_dict, info_dict)

            step_count += 1

            if all(term_dict.values()) or all(trunc_dict.values()):
                reason = "terminated" if all(term_dict.values()) else "truncated"
                print(f"\n  Episode ended early at step {step} ({reason})")
                self._print_step(step, obs_dict, reward_dict, info_dict)
                break



            self.env.render()
            time.sleep(1 / 60)

        return {"steps": step_count, "total_rewards": episode_rewards}

    # ------------------------------------------------------------------
    # Per-step console output
    # ------------------------------------------------------------------
    def _print_step(self, step, obs_dict, reward_dict, info_dict):
        print(f"\n  --- step {step} ---")
        for d in obs_dict:
            pos = obs_dict[d][:3]
            info = info_dict[d]
            print(
                f"  {d:8s} | "
                f"pos=({pos[0]:5.2f},{pos[1]:5.2f},{pos[2]:5.2f}) | "
                f"reward={reward_dict[d]:7.3f} | "
                f"dist={info.get('dist_to_target', 0.0):5.2f} | "
                f"reached={str(info.get('reached', False)):5s} | "
                f"wall={info.get('wall_collision', False)} | "
                f"obs={info.get('obstacle_collision', False)} | "
                f"drone={info.get('drone_collision', False)}"
            )

    # ------------------------------------------------------------------
    # Run all episodes
    # ------------------------------------------------------------------
    def run(self):
        try:
            for ep in range(self.n_episodes):
                stats = self._run_episode(ep)

                print(f"\n  Episode {ep + 1} summary:")
                print(f"    steps:         {stats['steps']}")
                for d, total_r in stats["total_rewards"].items():
                    print(f"    {d} total reward: {total_r:.3f}")

                time.sleep(1.0)

        finally:
            print("\nClosing environment.")
            self.env.close()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    tester = EnvGUITester(
        world_scale=2.0,
        num_drones=1,
        num_static_obstacles=3,
        num_dynamic_obstacles=0,
        max_steps=200000,
        n_episodes=10,
    )
    tester.run()