# -*- coding: utf-8 -*-
import os
from stable_baselines3 import SAC
import numpy as np
import json

from train.expert_blender import ExpertPolicy
from train.sac_training_with_expert import ExpertBlendPolicy
from wrappers.env_3d import DroneSwarm3D
from pathlib import Path
from envs.createObjects import EnvObject, Dynamic


class SingleEpisodeRunner:
    def __init__(
        self,
        n_drones: int,
        n_static_obstacles: int,
        n_dynamic_obstacles: int,
        lidar: bool,
        model_num: int,
        render_mode: str = None,
        world_scale: float = 5.0,
        max_steps: int = 100000,
    ):
        self.n_drones = n_drones
        self.n_static_obstacles = n_static_obstacles
        self.n_dynamic_obstacles = n_dynamic_obstacles
        self.lidar = lidar
        self.render_mode = render_mode
        self.world_scale = world_scale
        self.max_steps = max_steps
        self.model_num = model_num

        MODELS_DIR = "models"
        self.MODEL_PREFIX = "sac"

        PROJECT_ROOT = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        self.MODEL_NAME = f"{self.MODEL_PREFIX}_{self.model_num}_{self.n_drones}_{self.n_static_obstacles}_{self.n_dynamic_obstacles}"
        self.MODEL_PATH = os.path.join(PROJECT_ROOT, MODELS_DIR, f"{self.MODEL_NAME}")

        self.env = DroneSwarm3D(
            world_scale=self.world_scale,
            num_drones_per_swarm=self.n_drones,
            num_static_obstacles=self.n_static_obstacles,
            num_dynamic_obstacles=self.n_dynamic_obstacles,
            lidar=self.lidar,
            max_steps=self.max_steps,
            render_mode=self.render_mode,
        )

        # --- Uncomment to load scenarios ---
        # scenarios = self.get_ml_test_scenarios()
        # selected_scenario = "hallway"
        # object_configs = scenarios[selected_scenario]()

        self.obs_dict, self.info = self.env.reset()

        # --- Uncomment to apply scenario ---

        # self.apply_object_config(self.env, object_configs)

    def _load_model(self):
        model = SAC.load(
            self.MODEL_PATH,
            custom_objects={
                "policy_class": ExpertBlendPolicy,
                "replay_buffer_class": None,
            },
            print_system_info=True,
        )

        # Reattach runtime expert object used by ExpertBlendPolicy
        model.policy.expert = ExpertPolicy(
            world_scale=self.world_scale,
            use_lidar=self.lidar,
        )

        return model

    def run_episode(self, seed: int = None):
        model = self._load_model()

        # obs_dict, _ = self.env.reset(seed=seed)

        done = {a: False for a in self.env.possible_agents}
        trunc = {a: False for a in self.env.possible_agents}
        ep_len = 0

        # ---- episode counters ----
        reached_agents = set()
        collided_agents = set()
        wall_collision_agents = set()
        obstacle_collision_agents = set()
        drone_collision_agents = set()

        # optional: count total collision events (not just unique drones)
        wall_collision_events = 0
        obstacle_collision_events = 0
        drone_collision_events = 0

        while True:
            # Get deterministic actions
            act_dict = {
                a: model.predict(obs, deterministic=True)[0]
                for a, obs in self.obs_dict.items()
            }
            obs_dict, _, done, trunc, info = self.env.step(act_dict)

            obs_dict, rewards, done, trunc, info = self.env.step(act_dict)

            ep_len += 1
            self.env.render()

            # ---- update counters from info dict ----
            for agent in self.env.possible_agents:
                if agent not in info:
                    continue

                agent_info = info[agent]

                # ever reached during this episode
                if agent_info.get("reached", False):
                    reached_agents.add(agent)

                # ever collided during this episode
                if agent_info.get("collided", False):
                    collided_agents.add(agent)

                # collision types
                if agent_info.get("wall_collision", False):
                    wall_collision_agents.add(agent)

                if agent_info.get("obstacle_collision", False):
                    obstacle_collision_agents.add(agent)

                if agent_info.get("drone_collision", False):
                    drone_collision_agents.add(agent)

                # instantaneous collision events this step
                if agent_info.get("wall_collision_now", False):
                    wall_collision_events += 1

                if agent_info.get("obstacle_collision_now", False):
                    obstacle_collision_events += 1

                if agent_info.get("drone_collision_now", False):
                    drone_collision_events += 1

            if all(done[a] or trunc[a] for a in done):
                print(f"\nEpisode finished in {ep_len} steps.")
                print(f"Total drones: {self.env.total_num_drones}")

                print("\n--- Unique drones affected during episode ---")
                print(f"Reached target:              {len(reached_agents)}")
                print(f"Collided (any type):         {len(collided_agents)}")
                print(f"Wall collision:              {len(wall_collision_agents)}")
                print(f"Obstacle collision:          {len(obstacle_collision_agents)}")
                print(f"Drone-drone collision:       {len(drone_collision_agents)}")

                print("\n--- Collision events during episode ---")
                print(f"Wall collision events:       {wall_collision_events}")
                print(f"Obstacle collision events:   {obstacle_collision_events}")
                print(f"Drone-drone collision events:{drone_collision_events}")

                print("\n--- Per-agent summary ---")
                for agent in self.env.possible_agents:
                    if agent in info:
                        agent_info = info[agent]
                        print(
                            f"{agent}: "
                            f"reached={agent_info.get('reached', False)}, "
                            f"collided={agent_info.get('collided', False)}, "
                            f"wall={agent_info.get('wall_collision', False)}, "
                            f"obstacle={agent_info.get('obstacle_collision', False)}, "
                            f"drone={agent_info.get('drone_collision', False)}"
                        )
                break

        self.env.close()

    def safe_unit(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return np.zeros_like(v)
        return v / norm

    # Pre-set folder with configs
    def _configs_dir(self) -> Path:
        return Path(__file__).resolve().parents[2] / "interface" / "configs"

    # Load object configurations from JSON files
    def _load_object_config(self, config_name: str) -> list[dict]:
        path = self._configs_dir() / config_name
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")
        return data

    # Load different scenarios
    def load_wall_holes_config(self) -> list[dict]:
        return self._load_object_config("wall_holes.json")

    def load_two_walls_with_holes_config(self) -> list[dict]:
        return self._load_object_config("two_walls_with_holes.json")

    def load_hallway_config(self) -> list[dict]:
        return self._load_object_config("hallway.json")

    # Load different scenarios

    # Apply object configurations to the environment
    def apply_object_config(
        self, env: DroneSwarm3D, object_configs: list[dict]
    ) -> None:
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

    def get_ml_test_scenarios(self) -> dict[str, callable]:
        return {
            "wall_holes": self.load_wall_holes_config,
            "two_walls_with_holes": self.load_two_walls_with_holes_config,
            "hallway": self.load_hallway_config,
        }


if __name__ == "__main__":
    runner = SingleEpisodeRunner(
        model_num=29,
        n_drones=1,
        n_static_obstacles=0,
        n_dynamic_obstacles=0,
        lidar=True,
        render_mode="human",
    )
    runner.run_episode()
