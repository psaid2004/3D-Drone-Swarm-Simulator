from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

class EpisodeStatsTracker:
    def __init__(self, log_path: str = 'episode_stats.csv'):
        self.log_path: Path = Path(log_path)
        self.ep_idx = 0,
        self.ep_rewards = []
        self.collision_count = 0
        self.reached_total = 0
        self.wall_hits = 0
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.log_path.exists():

            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode",
                    "steps",
                    "collisions",
                    "goals_reached",
                    "walls_hits",
                    "mean_reward",
                ])


    def reset_episode(self) -> None:
        self.ep_rewards.clear()
        self.collision_count = 0
        self.reached_total = 0
        self.wall_hits = 0


    def record_step(
            self,
            reward_dict: dict,
            possible_agents: list[str],
            total_num_drones: int, step_state
    ) -> None:
        step_mean_reward = float(np.mean([reward_dict[a] for a in possible_agents[:total_num_drones]]))

        self.ep_rewards.append(step_mean_reward)

        self.collision_count += int(np.sum(step_state.collisions[:total_num_drones]))
        self.reached_total += int(np.sum(step_state.just_reached[:total_num_drones]))


    def write_episode(self, step_count: int) -> None:
        """ Write episode stats to csv """

        mean_reward = float(np.mean(self.ep_rewards)) if self.ep_rewards else 0.0

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.ep_idx,
                step_count,
                self.collision_count,
                self.reached_total,
                self.wall_hits,
                mean_reward,
            ])

        self.ep_idx += 1
        self.ep_rewards.clear()
