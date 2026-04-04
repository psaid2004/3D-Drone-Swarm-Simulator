# -*- coding: utf-8 -*-
# config/experiment.py
"""
Single source of truth for experiment-wide settings.

Design:
- This file defines DEFAULTS (constants).
- Training / evaluation scripts import these values.
- Scripts may override locally, but should NOT mutate this module.
"""

# =============================================================================
# Core environment defaults
# =============================================================================
DEFAULT_NUM_DRONES = 8
DEFAULT_NUM_OBSTACLES = 4
DEFAULT_MAX_STEPS = 510

DEFAULT_LIDAR = True
DEFAULT_RENDER_MODE = None  # e.g. None or "human"

# 3D world / motion scaling (used by wrappers/pz_parallel_3d.py)
DEFAULT_WORLD_SCALE = 2.0
DEFAULT_VEL_SCALE = 6.0
DEFAULT_TAKEOFF_STEPS = 40
DEFAULT_TAKEOFF_VZ = 1.2

# =============================================================================
# Swarm task defaults
# =============================================================================
DEFAULT_SHARED_TARGET = True
DEFAULT_SWARM_MAX_SEP = 2.0
DEFAULT_SWARM_MIN_SEP = 0.25
DEFAULT_TERMINATE_ON_COHESION_BREAK = False
DEFAULT_TERMINATE_ON_SEPARATION = False

# =============================================================================
# Lidar defaults (3D wrapper)
# =============================================================================
LIDAR_NUM_RAYS = 72
LIDAR_NUM_LAYERS = 5
LIDAR_RANGE = 6.0
LIDAR_VERTICAL_FOV_DEG = 45.0
LIDAR_HORIZONTAL_FOV_DEG = 360.0
LIDAR_NOISE_STD = 0.0

# defaults per script (train vs test)
LIDAR_SHOW_RAYS_TRAIN = False
LIDAR_SHOW_RAYS_TEST = True

# =============================================================================
# Paths / naming
# =============================================================================
MODEL_PREFIX = "ppo_"
MODELS_DIR = "models"  # relative to project root
