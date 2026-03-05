"""
This file defines DEFAULTS (constants). Training / evaluation scripts import these values.
"""
from typing import Tuple

# =============================================================================
# Core environment defaults
# =============================================================================
# constants used if curriculum not called, i.e. for test runs
DEFAULT_NUM_DRONES = 8
DEFAULT_NUM_OBSTACLES = 12
DEFAULT_N_STEPS = 512
DEFAULT_LIDAR = True
DEFAULT_RENDER_MODE = None  # if "human" -> shows environment


# =============================================================================
# World
# =============================================================================
WORLD_SCALE = 10.0          # spawn in [-WORLD_SCALE, WORLD_SCALE]
SPAWN_MARGIN = 6.0          # min distance between agent, obstacles, target spawn

# =============================================================================
# Goal
# =============================================================================
DEFAULT_SHARED_TARGET = True
GOAL_RADIUS = 1.0
GOAL_BONUS = 150
GOAL_WALL_MARGIN = 3.5  # Minimum distance between the wall and the goal

# =============================================================================
# Swarm cohesion / separation
# =============================================================================
DEFAULT_SWARM_MAX_SEP = 5.0
DEFAULT_SWARM_MIN_SEP = 0.25
DEFAULT_TERMINATE_ON_COHESION_BREAK = False
DEFAULT_TERMINATE_ON_SEPARATION = False

# =============================================================================
# Obstacles
# =============================================================================
OBSTACLE_RADIUS = 0.3
OBSTACLE_HEIGHT_RANGE = (0, WORLD_SCALE)
OBSTACLE_DYNAMIC = False
OBSTACLE_DRIFT_AMP = 0.8
OBSTACLE_DRIFT_OMEGA = 0.35

# =============================================================================
# LiDAR
# =============================================================================
LIDAR_NUM_RAYS = 24  # TODO INCREASE THIS LATER
LIDAR_NUM_LAYERS = 5
LIDAR_RANGE = 6.0  # meters ahead
LIDAR_VERTICAL_FOV_DEG = 45.0  # ±22.5° up/down
LIDAR_HORIZONTAL_FOV_DEG = 360.0  # full circle
LIDAR_NOISE_STD = 0.0
LIDAR_SHOW_RAYS_TRAIN = False
LIDAR_SHOW_RAYS_TEST = True

# =============================================================================
# Motion
# =============================================================================
VEL_SCALE = 15.0
TAKEOFF_STEPS = 60
TAKEOFF_VZ = 1.0
HOVER_BIAS_VZ = 0.0

# =============================================================================
# Collision
# =============================================================================
COLLISION_DIST = 0.0

# =============================================================================
# Reward - per-step shaping
# =============================================================================
PROG_COEF = 4.5
TIME_PEN = 0.05
CNTRL_PEN = 0.01
NEAR_OBS_PEN = 0.6

# =============================================================================
# Reward - swarm shaping
# =============================================================================
COHESION_PEN = 5.0
SEPARATION_PEN = 3.0

# =============================================================================
# Reward - terminal bonuses / penalties
# =============================================================================
REACH_BONUS = 150.0
TRUNC_PEN = 300.0
COLLISION_PEN = 30.0
ZERO_SPEED_PEN = 1.0

# =============================================================================
# Avoid division by zero
# =============================================================================
eps = 1e-8

# =============================================================================
# Paths
# =============================================================================
MODELS_DIR = "models"









