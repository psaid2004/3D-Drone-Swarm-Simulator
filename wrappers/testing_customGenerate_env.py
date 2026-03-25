import time
import numpy as np
from wrappers.env_3d import DroneSwarm3D
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def safe_unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.zeros_like(v)
    return v / norm

def main():
    # --- create environment ----
    n_drones = 5
    n_steps = 5000

    pz_env = DroneSwarm3D(
        world_scale = 4.0,
        num_drones_per_swarm=n_drones,
        num_static_obstacles = 2,
        num_dynamic_obstacles = 2,
        max_steps=n_steps,
        num_swarms=1,
        lidar=False,
        render_mode="human")

    obs_dict, info = pz_env.reset()

    print("Initial observations:")
    for drone, obs in obs_dict.items():
        print(drone, obs[:6])  # pos + vel
    print("Targets:", [info[drone].get("reached") for drone in obs_dict])

    print(n_steps)
    for step in range(n_steps):
        action_dict = {}
        for drone, obs in obs_dict.items():
            # Move roughly toward target + centroid
            to_target = obs[6:9]
            to_centroid = obs[12:15]
            dir_vec = 0.8 * safe_unit(to_target) + 0.2 * safe_unit(to_centroid)
            dir_vec = safe_unit(dir_vec)
            speed = np.clip(np.linalg.norm(to_target) / 5.0, 0.05, 1.0)
            u3 = speed * 2.0 - 1.0
            action_dict[drone] = np.array([dir_vec[0], dir_vec[1], dir_vec[2], u3], dtype=np.float32)

        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = pz_env.step(action_dict)

        # Print status every 50 steps
        if step % 50 == 0:
            print(f"Step {step}:")
            for d in obs_dict:
                print(f"  {d} pos: {obs_dict[d][:3]}, reward: {reward_dict[d]:.2f}, collided: {info_dict[d]['collided']}")


        print(len(pz_env._obstacles))
        # Stop if all done
        if all(term_dict.values()):
            print(f"Episode ended at step {step}")
            break

        pz_env.render()
        time.sleep(1 / 60)

    input("Press Enter to close environment...")
    pz_env.close()

if __name__ == "__main__":
    main()