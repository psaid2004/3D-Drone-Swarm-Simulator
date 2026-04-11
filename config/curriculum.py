# -*- coding: utf-8 -*-
"""
The following class contains the curriculum by which training is performed.
"""

from __future__ import annotations
from typing import List, Tuple


class Curriculum:
    """
    SB3 SAC curriculum (3D drone training)

    Each stage tuple:
    (
        world_scale,
        num_drones,
        num_static_obstacles,
        num_dynamic_obstacles,
        stage_training_timesteps,
        episode_horizon,
        lidar_enabled
    )

    Notes:
    - stage_training_timesteps = how long to train for in that curriculum stage
    - episode_horizon = per-episode max number of steps before truncation
    - lidar_enabled = whether LiDAR is included in the observation/action logic for that stage
    """

    def __init__(self):
        self.stages: List[Tuple[float, int, int, int, int, int, bool]] = [
            (3, 3, 0, 0, 300_000, 1500, True),
            (3, 5, 0, 0, 350_000, 1700, True),
            (3, 5, 2, 0, 400_000, 1700, True),
            (4, 5, 3, 0, 450_000, 2000, True),
            (4, 5, 4, 1, 450_000, 2200, True),
            (5, 5, 5, 1, 500_000, 2500, True),
            (5, 8, 5, 1, 500_000, 2700, True),
            (6, 8, 6, 2, 550_000, 3000, True),
            (7, 8, 8, 2, 550_000, 3300, True),
            (8, 10, 8, 3, 600_000, 3600, True),
            (9, 10, 10, 4, 700_000, 3900, True),
        ]
        self.current_stage = 0

    def get_stage(self):
        """Return the parameters of the current stage."""
        return self.stages[self.current_stage]

    def next_stage(self):
        """Move to the next stage if available."""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            print(f"Advancing to Curriculum Stage {self.current_stage}")
        else:
            print("Already at final stage.")
