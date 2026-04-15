import math
import pybullet as p
import pybullet_data
import random
import time
import numpy as np
from envs.createObjects import *
from pathlib import Path

SUPPORTED_SHAPES = ["box", "wall", "sphere", "cylinder"]


def generate_edge_xyz(edge: int, world_scale: float, margin: float, rng=None,) -> np.ndarray:
    """
    Return a random position on the given world edge.
    """
    if rng is None:
        rng = random
    s = world_scale
    m = margin
    if edge == 0:  # +X edge
        pos = [
            s - m,
            rng.uniform(-s + m, s - m),
            rng.uniform(m, s - m),
        ]
    elif edge == 1:  # -X edge
        pos = [
            -s + m,
            rng.uniform(-s + m, s - m),
            rng.uniform(m, s - m),
        ]
    elif edge == 2:  # +Y edge
        pos = [
            rng.uniform(-s + m, s - m),
            s - m,
            rng.uniform(m, s - m),
        ]
    else:  # -Y edge
        pos = [
            rng.uniform(-s + m, s - m),
            -s + m,
            rng.uniform(m, s - m),
        ]
    return np.array(pos, dtype=np.float32)

WORLD_BOUNDS = {
    "x" : (-3.0, 3.0),
    "y" : (-3.0, 3.0),
    "z" : (0, 5.0)
}
WORLD_BOUNDS = {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "z": (0, 5.0)}
