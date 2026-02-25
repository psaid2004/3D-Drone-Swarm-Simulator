"""
This file defines DEFAULTS (constants). Training / evaluation scripts import these values.
"""
from typing import Tuple

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

# =============================================================================
# Drone / obstacle geometry defaults
# =============================================================================
ARENA_SIZE = 100.0  # metres (box arena in 3D)

GOAL_WALL_MARGIN = 3.5  # Minimum distance between the wall and the goal
SUCCESS_DIST = 1.0  # Distance at which the drone is assumed to land goal
MIN_GOAL_DIST = 15.0
SPAWN_MARGIN = 6.0  # Minimum Distance Between agent spawn and another agent/obstacle

OBSTACLE_RADIUS = 0.30
OBSTACLE_HEIGHT_RANGE = (0.5, 1.8)
OBSTACLE_DYNAMIC = False
OBSTACLE_DRIFT_AMP = 0.8
OBSTACLE_DRIFT_OMEGA = 0.35

# =============================================================================
# Reward Functions Defaults
# =============================================================================

# thresholds
SAFE_DIST = 11.0
DANGEROUS_DIST = 8.0
FUNNEL_R = 15.0
BRAKE_R = 9.0
ALIGN_R = 15.0
MIN_PUSH_SPEED = 0.5

# core coefficients
PROG_COEF = 4.5
STEP_COST = 0.01
ACTION_PEN = 0.005
ALIGN_COEF = 0.5
BRAKE_COEF = 2.0
REP_COEF = 0.5
TIME_PEN = 0.05
NEAR_OBS_PEN = 0.6

# terminal Bonuses
COLLISION_PEN = 30.0
COLLISION_DIST = 0.12
REACH_BONUS = 150.0
TRUNC_PEN = 300.0
ZERO_SPEED_PEN = 1.0

# dodging bonuses
AWAY_DRONE_COEF = 2.5
AWAY_OBS_COEF = 1.5

# avoid division by zero
eps = 1e-8

# TODO GROUP CORRECTLY
COHESION_PEN = 50.0
SEPARATION_PEN = 30.0

# --- motion helpers ---
VEL_SCALE = 15.0,  # too small -> may not reach target
TAKEOFF_STEPS = 60
TAKEOFF_VZ = 1.0
HOVER_BIAS_VZ = 0.0

GOAL_RADIUS = 0 # TODO SET VALUES
GOAL_BONUS = 0
CNTRL_PEN = 0
