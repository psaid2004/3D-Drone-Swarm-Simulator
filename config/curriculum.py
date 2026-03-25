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
        lidar_enabled,
    )

    Notes:
    - stage_training_timesteps = how long to train in that curriculum stage
    - episode_horizon = per-episode max number of steps before truncation
    - lidar_enabled = whether LiDAR is included in the observation/action logic for that stage
    """

    def __init__(self):

        self.stages: List[Tuple[float, int, int, int, int, int, bool]] = [

            # ---- SINGLE DRONE ----
            (2, 1, 0, 0, 1_000_000, 1300, False),   # Stage 0: one drone, empty world
            (3, 1, 0, 0, 1_000_000, 1300, False),  # Stage 0: one drone, empty world
            (4, 1, 0, 0, 500_000, 1300, False),  # Stage 0: one drone, empty world
            (5, 1, 0, 0, 500_000, 1300, False),  # Stage 0: one drone, empty world

            (2, 1, 0, 0, 1_000_000, 1300, True),   # Stage 0: one drone, empty world
            (3, 1, 0, 0, 1_000_000, 1300, True),  # Stage 0: one drone, empty world
            (4, 1, 0, 0, 500_000, 1300, True),  # Stage 0: one drone, empty world
            (5, 1, 0, 0, 500_000, 1300, True),  # Stage 0: one drone, empty world





            (5, 1, 1, 0, 7_000_000, 1300, True),    # Stage 1: static obstacles
            (6, 1, 5, 0, 700_000, 1500, True),    # Stage 2: larger world
            (7, 1, 8, 0, 800_000, 1550, True),    # Stage 3
            (8, 1, 10, 0, 900_000, 1550, True),   # Stage 4

            # introduce dynamic obstacles
            (8, 1, 6, 2, 900_000, 550, True),    # Stage 5
            (9, 1, 8, 3, 900_000, 600, True),    # Stage 6
            (10, 1, 10, 4, 1_000_000, 650, True),  # Stage 7

            # ---- TWO DRONES ----
            (6, 2, 0, 0, 500_000, 500, False),   # Stage 8
            (6, 2, 3, 0, 600_000, 550, True),    # Stage 9
            (7, 2, 5, 0, 700_000, 550, True),    # Stage 10
            (8, 2, 6, 1, 800_000, 550, True),    # Stage 11
            (9, 2, 8, 2, 900_000, 600, True),    # Stage 12
            (10, 2, 10, 3, 900_000, 600, True),  # Stage 13

            # ---- FOUR DRONES ----
            (7, 4, 0, 0, 500_000, 600, False),   # Stage 14
            (8, 4, 3, 0, 500_000, 600, True),    # Stage 15
            (9, 4, 5, 1, 600_000, 650, True),    # Stage 16
            (10, 4, 7, 2, 800_000, 650, True),   # Stage 17
            (12, 4, 10, 3, 1_000_000, 750, True), # Stage 18

            # ---- FULL SWARM ----
            (10, 8, 6, 2, 1_400_000, 600, True),   # Stage 19
            (12, 8, 8, 3, 1_500_000, 700, True),   # Stage 20
            (14, 8, 10, 4, 1_700_000, 800, True),  # Stage 21
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