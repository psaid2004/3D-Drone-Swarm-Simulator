import numpy as np
from pettingzoo import ParallelEnv

from envs.drone_env import DroneWorldEnv
from envs.drone_env import MAX_V, MAX_A


def env(num_drones, num_obstacles, max_steps,
        lidar, render_mode="rgb_array", max_v=MAX_V, max_a=MAX_A, spawn_area=100.0):
    base = _DronePZ(num_drones, num_obstacles, max_steps,
                    lidar, render_mode, max_v, max_a,spawn_area=spawn_area)
    return base


class _DronePZ(ParallelEnv):
    metadata = {"name": "multi_drone_collision_avoidance_v0", "render_modes": ["human", "rgb_array"],
                "render_fps": 30, }

    def __init__(self,
                 num_drones: int,
                 num_obstacles: int,
                 max_steps: int,
                 lidar: bool,
                 render_mode: str,
                 max_v: float,
                 max_a: float,
                 spawn_area: float):
        super().__init__()

        # ---------- underlying Gymnasium env ------------------------------
        self.world = DroneWorldEnv(num_drones=num_drones,
                                   num_obstacles=num_obstacles,
                                   max_steps=max_steps,
                                   lidar=lidar,
                                   render_mode=render_mode,
                                   max_v=max_v, max_a=max_a,
                                   spawn_area=spawn_area)

        # ---------- static agent list -------------------------------------
        self.possible_agents = [f"drone_{i}" for i in range(num_drones)]
        self.agents = self.possible_agents[:]
        self.render_mode = render_mode
        # same obs / action space for every agent
        obs_space = self.world.observation_space
        act_space = self.world.action_space
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: act_space for a in self.possible_agents}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def seed(self, seed: int | None = None) -> list[int]:
        self.world.reset(seed=seed)  # reseeds RNG
        return [seed if seed is not None else
                int(self.world.np_random.integers(2 ** 32))]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # --------------- ParallelEnv API --------------------------------------
    def reset(self, seed=None, options=None):
        obs, info = self.world.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        return {a: obs[i] for i, a in enumerate(self.agents)}, info

    def step(self, action_dict):
        live_now = self.agents.copy()  # agents *before* step

        # build full (n,2) action array
        acts = np.zeros((self.world.n, 2), dtype=np.float32)
        for idx, name in enumerate(self.possible_agents):
            if name in action_dict:
                acts[idx] = action_dict[name]

        obs, rew, done, trunc, info = self.world.step(acts)

        # build dicts only for live_now agents
        obs_d = {a: obs[i] for i, a in enumerate(self.possible_agents) if a in live_now}
        rew_d = {a: rew[i] for i, a in enumerate(self.possible_agents) if a in live_now}
        done_d = {a: done[i] for i, a in enumerate(self.possible_agents) if a in live_now}
        trunc_d = {a: trunc[i] for i, a in enumerate(self.possible_agents) if a in live_now}
        info_d = {a: info[a] for a in live_now}
        # 2) Update self.agents: remove only truly done agents
        # copy every root key that starts with "episode_"
        if done.all() or trunc.any():  # ← last step of the episode
            # original names from DroneWorldEnv
            coll = info["collision_count"]
            goals = info["goals_reached"]
            wall_hits = info["wall_hits"]
            for a in live_now:  # add to every surviving agent
                info_d[a]["collision_count"] = coll
                info_d[a]["goals_reached"] = goals
                info_d[a]["wall_hits"] = wall_hits
        # remove finished/truncated agents *after* dicts are ready
        self.agents = [a for a in self.agents if not done_d.get(a, False) and not trunc_d.get(a, False)]
        # self.agents = [a for a in live_now if not (done_d[a] or trunc_d[a])]
        return obs_d, rew_d, done_d, trunc_d, info_d

    def change_spawn_area(self,area:float):
        if 0.0 < area <= 100:
            self.world.spawn_area = area

    # optional text‑mode render
    def render(self):
        # print("ParallelEnv render not implemented — use world.render()")

        return self.world.render(self.render_mode)

    def close(self):
        self.world.close()
