from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class StepState:
    time_limit: bool

    collisions: np.ndarray
    wall_collisions: np.ndarray
    obstacle_collisions: np.ndarray
    drone_collisions: np.ndarray

    all_collided: bool

    positions: np.ndarray
    max_sep: float
    min_sep: float
    too_far: bool
    too_close: bool

    reached_now: np.ndarray
    just_reached: np.ndarray

    no_active_drones_left: bool

    episode_done: bool

    cohesion_pen: float
    sep_pen: float