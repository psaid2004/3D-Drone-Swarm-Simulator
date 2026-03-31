import os
from stable_baselines3 import SAC
import numpy as np
from wrappers.env_3d import DroneSwarm3D

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

        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    def run_episode(self, seed: int = None):
        model = SAC.load(self.MODEL_PATH)
        obs_dict, _ = self.env.reset(seed=seed)

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
            # deterministic actions
            act_dict = {
                a: model.predict(obs, deterministic=True)[0]
                for a, obs in obs_dict.items()
            }

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


if __name__ == "__main__":
    runner = SingleEpisodeRunner(
        model_num=0,
        n_drones=8,
        n_static_obstacles=8,
        n_dynamic_obstacles=2,
        lidar=True,
        render_mode="human",
    )
    runner.run_episode()