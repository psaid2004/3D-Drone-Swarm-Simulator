# train/evaluation/test_run.py
import time
import numpy as np

from wrappers.pz_parallel_3d import env as make_env


NUM_DRONES = 2
NUM_OBSTACLES = 2
N_STEPS = 2000

LIDAR = True
DYNAMIC_OBSTACLES = True


def _safe_unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.zeros_like(v)
    return v / n


def test_run(
    n_drones: int = NUM_DRONES,
    n_obstacles: int = NUM_OBSTACLES,
    n_steps: int = N_STEPS,
):
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
        shared_target=True,
        swarm_max_sep=2.0,
        swarm_min_sep=0.25,
        terminate_on_cohesion_break=False,
        terminate_on_separation=False,
        # lidar config
        lidar_num_rays=72,
        lidar_num_layers=5,
        lidar_range=6.0,
        lidar_vertical_fov_deg=45.0,
        lidar_horizontal_fov_deg=360.0,
        lidar_noise_std=0.0,
        lidar_show_rays=True,
        world_scale=2,
        vel_scale=6.0,          # try 6–8
        takeoff_steps=40,       # optional: shorter takeoff
        takeoff_vz=1.2,         # optional: faster climb
    )

    obs_dict, info = pz_env.reset()

    for t in range(n_steps):
        if len(pz_env.agents) == 0:
            obs_dict, info = pz_env.reset()

        # Heuristic "go to target + stay together" controller (NOT RL)
        # Observation layout:
        # pos(0:3), vel(3:6), to_target(6:9), to_obs(9:12), to_centroid(12:15), to_nearest_teammate(15:18), lidar...
        action_dict = {}
        for a in pz_env.agents:
            obs = obs_dict[a]
            to_target = obs[6:9]
            to_centroid = obs[12:15]

            # move mostly to target, slightly to centroid (cohesion)
            dir_vec = 0.85 * _safe_unit(to_target) + 0.15 * _safe_unit(to_centroid)
            dir_vec = _safe_unit(dir_vec)

            # speed control: increase when far from target, slow down when close
            dist = float(np.linalg.norm(to_target))
            speed01 = np.clip(dist / 3.0, 0.2, 1.0)  # [0.2..1.0]
            u3 = float(speed01 * 2.0 - 1.0)          # map [0..1] -> [-1..1]

            action_dict[a] = np.array([dir_vec[0], dir_vec[1], dir_vec[2], u3], dtype=np.float32)

        obs_dict, reward_dict, term_dict, trunc_dict, info = pz_env.step(action_dict)
        pz_env.render()
        time.sleep(1 / 60)

    input("Press Enter to close...")
    pz_env.close()


if __name__ == "__main__":
    test_run()
