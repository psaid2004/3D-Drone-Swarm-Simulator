import time
import numpy as np
from wrappers.pz_parallel_3d import env as make_env

def safe_unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.zeros_like(v)
    return v / norm

def main():
    # --- create environment ---
    n_drones = 4
    n_obstacles = 2
    n_steps = 500

    pz_env = make_env(
        num_drones=n_drones,
        render_mode="human",
        num_static_obstacles = 2,
        num_dynamic_obstacles = 3,
        max_steps=n_steps,
        lidar=False,
    )

    obs_dict, info = pz_env.reset()

    print("Initial observations:")
    for drone, obs in obs_dict.items():
        print(drone, obs[:6])  # pos + vel
    print("Targets:", [info[drone].get("reached") for drone in obs_dict])

    for step in range(n_steps):
        action_dict = {}
        for drone, obs in obs_dict.items():
            # Move roughly toward target + centroid
            to_target = obs[6:9]
            to_centroid = obs[12:15]
            dir_vec = 0.8 * safe_unit(to_target) + 0.2 * safe_unit(to_centroid)
            dir_vec = safe_unit(dir_vec)
            speed = np.clip(np.linalg.norm(to_target) / 3.0, 0.2, 1.0)
            u3 = speed * 2.0 - 1.0
            action_dict[drone] = np.array([dir_vec[0], dir_vec[1], dir_vec[2], u3], dtype=np.float32)

        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = pz_env.step(action_dict)

        # Print status every 50 steps
        if step % 50 == 0:
            print(f"Step {step}:")
            for d in obs_dict:
                print(f"  {d} pos: {obs_dict[d][:3]}, reward: {reward_dict[d]:.2f}, collided: {info_dict[d]['collided']}")

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