# -*- coding: utf-8 -*-
import time
import numpy as np
from wrappers.env_3d import DroneSwarm3D


def safe_unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.zeros_like(v)
    return v / norm


def run_episode(env, n_steps):
    obs_dict, info = env.reset()

    print("\n===== NEW EPISODE STARTED =====")
    print("Initial observations:")

    for step in range(n_steps):
        action_dict = {}

        for drone, obs in obs_dict.items():
            to_target = obs[6:9]
            to_centroid = obs[12:15]

            dir_vec = 0.8 * safe_unit(to_target) + 0.2 * safe_unit(to_centroid)
            dir_vec = safe_unit(dir_vec)

            speed = np.clip(np.linalg.norm(to_target) / 5.0, 0.05, 1.0)
            u3 = speed * 2.0 - 1.0

            action_dict[drone] = np.array(
                [dir_vec[0], dir_vec[1], dir_vec[2], u3], dtype=np.float32
            )

        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)

        if step % 50 == 0:
            print(f"\n===== STEP {step} =====")

            for d in obs_dict:
                pos = obs_dict[d][:3]
                r = reward_dict[d]

                collided = info_dict[d]["collided"]
                reached = info_dict[d].get("reached", False)

                wall = info_dict[d].get("wall_collision", False)
                obs = info_dict[d].get("obstacle_collision", False)
                drone = info_dict[d].get("drone_collision", False)

                print(
                    f"{d:8s} | "
                    f"pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:6.2f}) | "
                    f"reward={r:7.2f} | "
                    f"collided={str(collided):5s} | "
                    f"reached={str(reached):5s} | "
                    f"wall={wall} obs={obs} drone={drone}"
                )

        if all(term_dict.values()):
            print(f"Episode ended early at step {step}")
            break

        env.render()
        time.sleep(1 / 60)


def main():
    n_drones = 2
    n_steps = 100

    pz_env = DroneSwarm3D(
        world_scale=5.0,
        num_drones_per_swarm=n_drones,
        num_static_obstacles=0,
        num_dynamic_obstacles=0,
        max_steps=n_steps,
        num_swarms=2,
        lidar=False,
        render_mode="human",
    )

    # Run multiple episodes to test reset
    n_episodes = 5

    for ep in range(n_episodes):
        print("\n==============================")
        print(f"RUNNING EPISODE {ep + 1}")
        print("==============================")

        run_episode(pz_env, n_steps)

        print("\nReset completed successfully.\n")
        time.sleep(2)

    input("Press Enter to close environment...")
    pz_env.close()


if __name__ == "__main__":
    main()
