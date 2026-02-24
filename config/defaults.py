"""
This file defines DEFAULTS (constants). Training / evaluation scripts import these values.
"""

# =============================================================================
# Core environment defaults
# =============================================================================
# constants used if curriculum not called, i.e. for test runs
DEFAULT_NUM_DRONES = 2
DEFAULT_NUM_OBSTACLES = 0
DEFAULT_N_STEPS = 512

DEFAULT_LIDAR = True
DEFAULT_RENDER_MODE = None  # if "human" -> shows environment

# =============================================================================
# Swarm defaults
# =============================================================================
DEFAULT_SHARED_TARGET = True
DEFAULT_SWARM_MAX_SEP = 2.0
DEFAULT_SWARM_MIN_SEP = 0.25
DEFAULT_TERMINATE_ON_COHESION_BREAK = False
DEFAULT_TERMINATE_ON_SEPARATION = False

# =============================================================================
# Lidar defaults
# =============================================================================
LIDAR_NUM_RAYS = 24  # TODO INCREASE THIS LATER
LIDAR_NUM_LAYERS = 5
LIDAR_RANGE = 6.0  # meters ahead
LIDAR_VERTICAL_FOV_DEG = 45.0  # ±22.5° up/down
LIDAR_HORIZONTAL_FOV_DEG = 360.0  # full circle
LIDAR_NOISE_STD = 0.0

# defaults per script (train vs test)
LIDAR_SHOW_RAYS_TRAIN = False
LIDAR_SHOW_RAYS_TEST = True
