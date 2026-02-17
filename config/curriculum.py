"""config/curriculum.py

Central place for all curriculum definitions.

Why this exists:
- The project previously had multiple curriculum definitions spread across files.
- Keeping them in one place makes it easier to tweak training stages without
  touching the training logic.

IMPORTANT:
- Values below are copied from the *current* working setup to preserve behavior.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# =============================================================================
# SB3 PPO curriculum (CURRENT 3D TRAINING)
# =============================================================================
# Each stage tuple:
#   (world_scale, active_drones, num_obstacles, timesteps)
#
# These values are taken 1:1 from train/sb_ppo_training.py (before refactor).
PPO_CURRICULUM: List[Tuple[float, int, int, int]] = [
    (2.0, 1, 10, 200_000),
    (2.5, 1, 8,  400_000),
    (3.0, 1, 6,  500_000),
    (3.5, 2, 6,  600_000),
    (4.5, 2, 8,  800_000),
]

# =============================================================================
# LEGACY staged curriculum (kept for backwards compatibility)
# =============================================================================
# This dict was previously in train/staged_curriculums.py.
# It is still exported as `CURRICULUM` so old imports keep working.
CURRICULUM: Dict[int, List[Tuple[float, int, int, int]]] = {
    1: [
        (100, 1, 0, 2_000_000),
        (40,  1, 3,   700_000),
        (40,  1, 5,   700_000),
        (60,  1, 8,   700_000),
        (100, 1, 10, 2_000_000),
    ],
    2: [
        (40, 2, 0, 1_000_000),
        (40, 2, 1,   700_000),
        (40, 2, 2,   700_000),
        (40, 2, 3,   700_000),
        (60, 2, 10, 2_000_000),
    ],
    4: [
        (50, 4, 0, 1_000_000),
        (60, 4, 2, 1_000_000),
        (60, 4, 4, 1_000_000),
        (80, 4, 8, 1_000_000),
        (80, 4, 10, 2_000_000),
    ],
    6: [
        (50,  6, 0, 1_000_000),
        (60,  6, 2, 1_000_000),
        (70,  6, 4, 1_500_000),
        (80,  6, 8, 2_000_000),
        (100, 6, 10, 3_000_000),
    ],
}
