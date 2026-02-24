# train/evaluation/test_run.py
"""Interactive smoke-test for the 3D PettingZoo wrapper.

Defaults come from config/defaults.py
Run:
  python -m train.evaluation.test_run
"""

import time
import numpy as np

from config.defaults import (
    DEFAULT_NUM_DRONES,
    DEFAULT_NUM_OBSTACLES,
    DEFAULT_MAX_STEPS,
    DEFAULT_LIDAR,
    DEFAULT_WORLD_SCALE,
    DEFAULT_VEL_SCALE,
    DEFAULT_TAKEOFF_STEPS,
    DEFAULT_TAKEOFF_VZ,
    DEFAULT_SHARED_TARGET,
    DEFAULT_SWARM_MAX_SEP,
    DEFAULT_SWARM_MIN_SEP,
    DEFAULT_TERMINATE_ON_COHESION_BREAK,
    DEFAULT_TERMINATE_ON_SEPARATION,
    LIDAR_NUM_RAYS,
    LIDAR_NUM_LAYERS,
    LIDAR_RANGE,
    LIDAR_VERTICAL_FOV_DEG,
    LIDAR_HORIZONTAL_FOV_DEG,
    LIDAR_NOISE_STD,
    LIDAR_SHOW_RAYS_TEST,
)

from wrappers.pz_parallel_3d import env as make_env

# =============================================================================
# DEFAULTS (overridable)
# =============================================================================
NUM_DRONES = DEFAULT_NUM_DRONES
NUM_OBSTACLES = DEFAULT_NUM_OBSTACLES
N_STEPS = 2000

LIDAR = DEFAULT_LIDAR
DYNAMIC_OBSTACLES = True


def _safe_unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.zeros_like(v)
    return v / n


def test_run(n_drones: int = NUM_DRONES, n_obstacles: int = NUM_OBSTACLES, n_steps: int = N_STEPS):
    pz_env = make_env(
        num_drones=n_drones,
        render_mode="human",
        num_obstacles=n_obstacles,
        max_steps=n_steps,
        lidar=LIDAR,
        obstacle_radius=0.30,
        obstacle_dynamic=DYNAMIC_OBSTACLES,
        debug=False,

        # swarm + shared target
        shared_target=DEFAULT_SHARED_TARGET,
        swarm_max_sep=DEFAULT_SWARM_MAX_SEP,
        swarm_min_sep=DEFAULT_SWARM_MIN_SEP,
        terminate_on_cohesion_break=DEFAULT_TERMINATE_ON_COHESION_BREAK,
        terminate_on_separation=DEFAULT_TERMINATE_ON_SEPARATION,

        # lidar config
        lidar_num_rays=LIDAR_NUM_RAYS,
        lidar_num_layers=LIDAR_NUM_LAYERS,
        lidar_range=LIDAR_RANGE,
        lidar_vertical_fov_deg=LIDAR_VERTICAL_FOV_DEG,
        lidar_horizontal_fov_deg=LIDAR_HORIZONTAL_FOV_DEG,
        lidar_noise_std=LIDAR_NOISE_STD,
        lidar_show_rays=LIDAR_SHOW_RAYS_TEST,

        # world / motion scaling
        world_scale=DEFAULT_WORLD_SCALE,
        vel_scale=DEFAULT_VEL_SCALE,
        takeoff_steps=DEFAULT_TAKEOFF_STEPS,
        takeoff_vz=DEFAULT_TAKEOFF_VZ,
    )

    obs_dict, _ = pz_env.reset()

    for _t in range(n_steps):
        if len(pz_env.agents) == 0:
            obs_dict, _ = pz_env.reset()

        action_dict = {}
        for a in pz_env.agents:
            obs = obs_dict[a]
            to_target = obs[6:9]
            to_centroid = obs[12:15]

            dir_vec = 0.85 * _safe_unit(to_target) + 0.15 * _safe_unit(to_centroid)
            dir_vec = _safe_unit(dir_vec)

            dist = float(np.linalg.norm(to_target))
            speed01 = np.clip(dist / 3.0, 0.2, 1.0)
            u3 = float(speed01 * 2.0 - 1.0)

            action_dict[a] = np.array([dir_vec[0], dir_vec[1], dir_vec[2], u3], dtype=np.float32)

        obs_dict, _reward_dict, _term_dict, _trunc_dict, _info = pz_env.step(action_dict)
        pz_env.render()
        time.sleep(1 / 60)

    input("Press Enter to close...")
    pz_env.close()


if __name__ == "__main__":
    test_run()
