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

        while True:
            # Get deterministic actions
            act_dict = {a: model.predict(obs, deterministic=True)[0]
                        for a, obs in obs_dict.items()}
            obs_dict, _, done, trunc, info = self.env.step(act_dict)

            # Print speed of each drone
            # for i, agent in enumerate(self.env.possible_agents):
            #     if i < self.env.total_num_drones:
            #         vel = self.env._env._getDroneStateVector(i)[10:13]
            #         speed = np.linalg.norm(vel)
            #         print(f"{agent} speed: {speed:.3f}")

            ep_len += 1
            self.env.render()

            if all(done[a] or trunc[a] for a in done):
                sample_agent = next(iter(info))
                stats = info[sample_agent]
                print(f"Episode finished in {ep_len} steps.")
                print(f"Collisions: {stats.get('collided', 0)}")
                print(f"Goals reached: {stats.get('reached', 0)}")
                print(f"Wall hits: {stats.get('wall_collision', 0)}")
                break

        self.env.close()


if __name__ == "__main__":
    runner = SingleEpisodeRunner(
        model_num=2,
        n_drones=1,
        n_static_obstacles=0,
        n_dynamic_obstacles=0,
        lidar=True,
        render_mode="human",
    )
    runner.run_episode()