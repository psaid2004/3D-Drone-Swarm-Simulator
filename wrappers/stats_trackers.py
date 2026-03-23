from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

class EpisodeStatsTracker:
    def __init__(self, log_path: str = 'episode_stats.csv'):
        self.log_path: Path = Path(log_path)
        self.ep_idx = 0
        self.ep_rewards = []
        self.total_collision_count = 0
        self.reached_total = 0
        self.wall_collisions = 0
        self.drone_collisions = 0
        self.obstacle_collisions = 0
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.log_path.exists():

            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode",
                    "steps",
                    "total_collision_counts",
                    "reached_total"
                    "wall_collisions"
                    "drone_collisions",
                    "obstacle_collisions",
                    "mean_reward",
                ])


    def reset_episode(self) -> None:
        self.ep_rewards.clear()
        self.total_collision_count = 0
        self.reached_total = 0
        self.wall_collisions = 0
        self.drone_collisions = 0
        self.obstacle_collisions = 0


    def record_step(
            self,
            reward_dict: dict,
            possible_agents: list[str],
            total_num_drones: int,
            step_state,
            active_mask: np.ndarray,
    ) -> None:
        # only consider drones that contributed to the episode (not previously dead or despawned drones)
        contributing_drones = []

        for i, a in enumerate(possible_agents[:total_num_drones]):
            if active_mask[i] or step_state.reached_now[i] or step_state.collisions[i]:
                contributing_drones.append(reward_dict[a])

        step_mean_reward = float(np.mean(contributing_drones) if contributing_drones else 0.0)
        self.ep_rewards.append(step_mean_reward)

        self.total_collision_count += int(np.sum(step_state.collisions[:total_num_drones]))
        self.wall_collisions += int(np.sum(step_state.wall_collisions[:total_num_drones]))
        self.drone_collisions += int(np.sum(step_state.drone_collisions[:total_num_drones]))
        self.obstacle_collisions += int(np.sum(step_state.obstacle_collisions[:total_num_drones]))

        self.reached_total += int(np.sum(step_state.reached_now[:total_num_drones]))


    def write_episode(self, step_count: int) -> None:
        """ Write episode stats to csv """

        mean_reward = float(np.mean(self.ep_rewards)) if self.ep_rewards else 0.0

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.ep_idx,
                step_count,
                self.total_collision_count,
                self.reached_total,
                self.wall_collisions,
                self.drone_collisions,
                self.obstacle_collisions,
                mean_reward,
            ])

        self.ep_idx += 1
        self.ep_rewards.clear()
