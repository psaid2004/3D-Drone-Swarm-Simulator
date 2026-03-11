"""
This file defines DEFAULTS (constants). Training / evaluation scripts import these values.
"""
# =============================================================================
# DEFAULT OBSERVATION SLICES
# =============================================================================
POS_SLICE = slice(0, 3) # position vector
VEL_SLICE = slice(3, 6) # velocity vector
TARGET_SLICE = slice(6, 9) # vector to target
OBS_SLICE = slice(9, 12) # vector to nearest obstacle
CENTROID_SLICE = slice(12, 15) # vector to nearest centroid
TEAMMATE_SLICE = slice(15, 18) # vector to nearest teammate

# =============================================================================
# Swarm Spawn Variables
# =============================================================================
SWARM_RADIUS = 0.6 # Radius of swarm (AKA how far from the center can drones spawn)
SWARM_EDGE_MARGIN = 1  # How far from environment border the swarm center is
SWARM_MIN_SEPARATION = 0.15  # Min distance between drones in the swarm

# =============================================================================
# Goal
# =============================================================================
GOAL_RADIUS = 1 # actual radius of the goal
#TODO USE THIS FOR SPAWNING TARGETS
GOAL_EDGE_MARGIN = 3.5  # Minimum distance between the wall and the goal
GOAL_REACH_DISTANCE = 0.9 # How close drone needs to be around goal to be a successful reach

# =============================================================================
# Swarm cohesion / separation
# =============================================================================
DEFAULT_SWARM_MAX_SEP = 5.0
DEFAULT_SWARM_MIN_SEP = 0.25
DEFAULT_TERMINATE_ON_COHESION_BREAK = False
DEFAULT_TERMINATE_ON_SEPARATION = False
# TODO make sure this works well with the reward function

# =============================================================================
# Obstacles
# =============================================================================
MIN_SPAWN_DIST_FROM_DRONE = 0.5
SUPPORTED_SHAPES = ["box", "wall", "sphere", "cylinder"]

#TODO MAKE SURE THE LIDAR ACTUALLY WORKS (DAVID)

# =============================================================================
# LiDAR
# =============================================================================
LIDAR_NUM_RAYS = 24
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
# TAKEOFF_STEPS = 30 # first 30 steps of episode is purely drone takeoff
# TAKEOFF_VZ = 1.0 # takeoff velocity of drone during first takeoff steps
# HOVER_BIAS_VZ = 0.0 # add constant upward bias to counter gravity drift

# =============================================================================
# Collision
# =============================================================================
COLLISION_DIST = 0.0

# =============================================================================
# Reward - per-step shaping
# =============================================================================
PROG_COEF = 4.5 # reward coeff for progressing closer to goal
TIME_PEN = 0.05 # small time penalty each step
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
SHARED_COLLISION_PEN = 30.0 # small group penalty if any collision occurs
WALL_COLLISION_PEN = 30.0 # drone vs wall collision penalty
OBSTACLE_COLLISION_PEN = 30.0 # drone vs obstacle collision penalty
DRONE_COLLISION_PEN = 30.0 # drone vs drone collision penalty
ZERO_SPEED_PEN = 1.0
GOAL_BONUS = 150 # one-time bonus for reaching the goal

# =============================================================================
# Avoid division by zero
# =============================================================================
eps = 1e-8

# =============================================================================
# Paths
# =============================================================================
MODELS_DIR = "models"









