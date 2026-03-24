import time
import numpy as np
from wrappers.env_3d import DroneSwarm3D
from pathlib import Path
import json
from envs.createObjects import EnvObject, Dynamic


def safe_unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.zeros_like(v)
    return v / norm

# Pre-set folder with configs
def _configs_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "interface" / "configs"

# Load object configurations from JSON files
def _load_object_config(config_name: str) -> list[dict]:
    path = _configs_dir() / config_name
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")
    return data

# Load different scenarios
def load_wall_holes_config() -> list[dict]:
    return _load_object_config("wall_holes.json")


def load_two_walls_with_holes_config() -> list[dict]:
    return _load_object_config("two_walls_with_holes.json")


def load_hallway_config() -> list[dict]:
    return _load_object_config("hallway.json")
# Load different scenarios

# Apply object configurations to the environment
def apply_object_config(env: DroneSwarm3D, object_configs: list[dict]) -> None:
    """Spawn obstacles from JSON config into the environment."""
    env.remove_all_obstacles()
    env._obstacles = []

    static_count = 0
    dynamic_count = 0

    for cfg in object_configs:
        pos = [float(cfg["x"]), float(cfg["y"]), float(cfg["z"])]
        ori = [0.0, 0.0, float(cfg.get("yaw", 0.0))]
        size = max(0.1, float(cfg.get("size", 1.0)))
        shape = str(cfg.get("shape", "box"))

        if cfg.get("kind") == "dynamic":
            axis = int(cfg.get("axis", 0))
            min_pos = float(cfg.get("min", -1.0))
            max_pos = float(cfg.get("max", 1.0))
            if min_pos > max_pos:
                min_pos, max_pos = max_pos, min_pos
            speed = max(0.0, float(cfg.get("speed", 0.2)))
            direction = -1 if int(cfg.get("direction", 1)) < 0 else 1
            obj = Dynamic(axis, min_pos, max_pos, speed, direction, pos, size, ori)
            dynamic_count += 1
        else:
            obj = EnvObject(pos, size, ori)
            static_count += 1

        obj.create(shape, physics_call=env._call_physics)
        env._obstacles.append(obj)

    env.num_static_obstacles = static_count
    env.num_dynamic_obstacles = dynamic_count


def get_ml_test_scenarios() -> dict[str, callable]:
    return {
        "wall_holes": load_wall_holes_config,
        "two_walls_with_holes": load_two_walls_with_holes_config,
        "hallway": load_hallway_config,
    }

def main():
    # --- create environment ----
    n_drones = 1
    n_steps = 8000

    pz_env = DroneSwarm3D(
        world_scale = 5.0,
        num_drones_per_swarm=n_drones,
        num_static_obstacles = 0,
        num_dynamic_obstacles = 0,
        max_steps=n_steps,
        num_swarms=1,
        lidar=True,
        render_mode="human")

    # --- Uncomment to load scenarios ---
    scenarios = get_ml_test_scenarios()
    selected_scenario = "hallway"
    object_configs = scenarios[selected_scenario]()

    obs_dict, info = pz_env.reset()
    
    # --- Uncomment to apply scenario ---
    
    apply_object_config(pz_env, object_configs)


    print("Initial observations:")
    # for drone, obs in obs_dict.items():
    #     print(drone, obs[:6])  # pos + vel
    print("Targets:", [info[drone].get("reached") for drone in obs_dict])

    # print(n_steps)
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

        # print(len(pz_env._obstacles))
        # Stop if all done
        if all(term_dict.values()):
            print(f"Episode ended at step {step}")
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
            break

        pz_env.render()
        time.sleep(1 / 60)

    input("Press Enter to close environment...")
    pz_env.close()

if __name__ == "__main__":
    main()