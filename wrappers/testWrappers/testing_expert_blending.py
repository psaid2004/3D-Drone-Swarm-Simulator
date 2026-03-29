import time
import numpy as np
from wrappers.env_3d import DroneSwarm3D
from config.defaults import LIDAR_NUM_RAYS, LIDAR_NUM_LAYERS

from train.expert_blender import ExpertPolicy


def main():
    n_drones = 2
    n_static_obstacles = 10
    n_dynamic_obstacles = 5
    n_steps = 8000
    use_lidar = True
    num_swarms = 1
    world_scale = 3.0

    env = DroneSwarm3D(
        world_scale=world_scale,
        num_drones_per_swarm=n_drones,
        num_static_obstacles=n_static_obstacles,
        num_dynamic_obstacles=n_dynamic_obstacles,
        max_steps=n_steps,
        num_swarms=num_swarms,
        lidar=use_lidar,
        render_mode="human",
    )

    expert = ExpertPolicy(
        world_scale=world_scale,
        use_lidar=use_lidar,
    )

    obs_dict, info = env.reset()

    print("Initial observations:")
    for drone, obs in obs_dict.items():
        print(f"  {drone}: pos={obs[:3]}")

    for step in range(n_steps):
        action_dict = {
            drone: expert.action(obs)
            for drone, obs in obs_dict.items()
        }

        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)

        if step % 50 == 0:
            print(f"\n===== STEP {step} =====")
            for d in obs_dict:
                pos = obs_dict[d][:3]
                r = reward_dict[d]
                collided = info_dict[d].get("collided", False)
                reached = info_dict[d].get("reached", False)
                wall = info_dict[d].get("wall_collision", False)
                obs_hit = info_dict[d].get("obstacle_collision", False)
                drone_hit = info_dict[d].get("drone_collision", False)
                print(
                    f"  {d:8s} | "
                    f"pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:6.2f}) | "
                    f"reward={r:7.2f} | "
                    f"collided={str(collided):5s} | "
                    f"reached={str(reached):5s} | "
                    f"wall={wall} obs={obs_hit} drone={drone_hit}"
                )

        if all(term_dict.values()):
            print(f"Episode ended at step {step}")
            for d in obs_dict:
                pos = obs_dict[d][:3]
                r = reward_dict[d]
                collided = info_dict[d].get("collided", False)
                reached = info_dict[d].get("reached", False)
                wall = info_dict[d].get("wall_collision", False)
                obs_hit = info_dict[d].get("obstacle_collision", False)
                drone_hit = info_dict[d].get("drone_collision", False)
                print(
                    f"  {d:8s} | "
                    f"pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:6.2f}) | "
                    f"reward={r:7.2f} | "
                    f"collided={str(collided):5s} | "
                    f"reached={str(reached):5s} | "
                    f"wall={wall} obs={obs_hit} drone={drone_hit}"
                )
            break

        env.render()
        time.sleep(1 / 60)

    input("Press Enter to close...")
    env.close()


if __name__ == "__main__":
    main()