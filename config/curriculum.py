"""
The following class contains the curriculum by which training is performed.
"""

from __future__ import annotations
from typing import Dict, List, Tuple


class Curriculum:
    # SB3 SAC curriculum (CURRENT 3D TRAINING)
    # Each stage tuple: (world_scale (meters), active_drones, num_obstacles, time steps)

    def __init__(self):
        # Define the stages of the curriculum
        self.stages: List[Tuple[float, int, int, int]] = [
            (5, 1, 0, 200_000),  # Stage 0: one drone, no obstacles
            (5, 1, 3, 500_000),  # Stage 1: add a few obstacles
            # (6, 1, 5, 500_000),  # Stage 2: larger world, more obstacles
            # (7, 1, 8, 600_000),  # Stage 3: larger world, more obstacles
            # (8, 1, 10, 800_000),  # Stage 4: larger world, more obstacles
            # (6, 2, 0, 300_000),  # Stage 5: two drones, no obstacles
            # (6, 2, 2, 400_000),  # Stage 6: two drones, more obstacles
            # (7, 2, 4, 500_000),  # Stage 7: larger world, more obstacles
            # (7, 2, 5, 700_000),  # Stage 8: two drones, same obstacles
            # (8, 2, 6, 700_000),  # Stage 9: larger world, more obstacles
            # (10, 2, 8, 1_000_000),
            # (7, 4, 0, 400_000),
            # (8, 4, 2, 500_000),
            # (9, 4, 4, 700_000),
            # (10, 4, 6, 800_000),
            # (12, 4, 10, 1_200_000),
            # (10, 4, 8, 1_000_000),
            # (12, 8, 12, 1_500_000),
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