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
GOAL_RADIUS = 0.5 # actual radius of the goal
GOAL_EDGE_MARGIN = 0.6  # Minimum distance between the wall and the goal
GOAL_REACH_DISTANCE = 0.6 #0.4 # How close drone needs to be around goal to be a successful reach
COLORS =  [
    [0.0, 1.0, 0.0, 0.9],  # green
    [0.0, 0.0, 1.0, 0.9],  # blue
    [1.0, 1.0, 0.0, 0.9],  # yellow
    [1.0, 0.0, 1.0, 0.9],  # magenta
    [0.0, 1.0, 1.0, 0.9],  # cyan
    [1.0, 0.5, 0.0, 0.9],  # orange
    [0.5, 0.0, 1.0, 0.9],  # purple
]

# =============================================================================
# Swarm cohesion / separation
# =============================================================================
DEFAULT_SWARM_MAX_SEP = 5.0
DEFAULT_SWARM_MIN_SEP = 0.25 # TODO PEDRO THERE IS A SWARM_MIN_SEPARATION ABOVE BUT IT IS DEFINING THE SAME THING HERE?

# =============================================================================
# Obstacles
# =============================================================================
MIN_SPAWN_DIST_FROM_DRONE = 1.0
SUPPORTED_SHAPES = ["box", "wall", "sphere", "cylinder"]

# =============================================================================
# LiDAR
# =============================================================================
LIDAR_NUM_RAYS = 24
LIDAR_NUM_LAYERS = 3
LIDAR_RANGE = 3.0  # meters ahead
LIDAR_VERTICAL_FOV_DEG = 30.0  # ±22.5° up/down
LIDAR_HORIZONTAL_FOV_DEG = 359.0  # full circle
LIDAR_NOISE_STD = 0.0
LIDAR_SHOW_RAYS_TRAIN = False
LIDAR_SHOW_RAYS_TEST = False

# =============================================================================
# Collision
# =============================================================================
COLLISION_DIST = 0.0

# =============================================================================
# Reward - per-step shaping
# =============================================================================
PROG_COEF = 6.5 # reward coefficient for progressing closer to goal
TIME_PEN = 0.02 # small time penalty each step (until drone reaches or collides)
CNTRL_PEN = 0.001 # penalty coefficient for large / aggressive actions (prevents jittering)

NEAR_OBS_PEN = 0.6 # penalty for being too close to an obstacle
NEAR_FRONT_OBS_PEN = 0.6 # penalty for being too close to an obstacle in front of drone
NEAR_SIDE_OBS_PEN = 0.35 # penalty for being too close to an obstacle on side of drone
DIST_PEN = 0.5 #0.15 # penalty coefficient for staying far from target (prevents drone not moving) # TODO OTHER IMPLEMENTATIONS HAVE THIS AS 1
ZERO_SPEED_PEN = 1.0 # penalty for drone not moving
ALIGN_COEF = 0.5

# =============================================================================
# Reward - swarm shaping
# =============================================================================
COHESION_PEN = 5.0 # penalty for when drones in swarm spread too far
SEPARATION_PEN = 3.0 # penalty for when drones in swarm get too close

# =============================================================================
# Reward - terminal bonuses / penalties
# =============================================================================
WALL_COLLISION_PEN = 30.0 # drone vs wall collision penalty
OBSTACLE_COLLISION_PEN = 30.0 # drone vs obstacle collision penalty
DRONE_COLLISION_PEN = 30.0 # drone vs drone collision penalty
GOAL_BONUS = 150 # one-time bonus for reaching the goal
TRUNC_PEN = 200 # penalty for if episode was truncated

# =============================================================================
# Reward - smoothness coefficients
# =============================================================================
FINAL_MILE_R = 3.0
FINAL_MILE_COEF = 2.0

# =============================================================================
# Avoid division by zero
# =============================================================================
eps = 1e-8

# =============================================================================
# Paths
# =============================================================================
MODELS_DIR = "models"
























# OLD VARIABLE STORAGE
# TAKEOFF_STEPS = 30 # first 30 steps of episode is purely drone takeoff
# TAKEOFF_VZ = 1.0 # takeoff velocity of drone during first takeoff steps
# HOVER_BIAS_VZ = 0.0 # add constant upward bias to counter gravity drift
# DEFAULT_TERMINATE_ON_COHESION_BREAK = False
# DEFAULT_TERMINATE_ON_SEPARATION = False






