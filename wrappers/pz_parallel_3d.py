from __future__ import annotations

import csv
from typing import Dict, Optional, Tuple, List
import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
from pathlib import Path

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from config.defaults import *

class DroneSwarm3D(ParallelEnv):
    metadata = {"name": "drone_swarms_3d"}

    # ------ Initialisation ------
    def __init__(self, num_drones: int = 4, active_drones: Optional[int] = None, num_obstacles: int = 5,
                 lidar: bool = False, max_steps: int = 1000, render_mode: Optional[str] = None, world_scale: float = 10.0,
                 seed: Optional[int] = None):

        super().__init__()

        self.num_drones = int(num_drones)
        self.active_drones = int(active_drones) if active_drones is not None else self.num_drones
        self.active_drones = max(1, min(self.active_drones, self.num_drones))

        self.num_obstacles = int(num_obstacles)
        self.lidar = bool(lidar)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.world_scale = float(world_scale)

        self.dead = np.zeros(self.num_drones, dtype=np.bool_)
        self.wall_hits = 0
        self.ep_done = False
        self.ep_idx = 0

        self.rng = np.random.default_rng(seed) # this is np_random from 2D

        self.possible_agents = [f"drone_{i}" for i in range(self.num_drones)]
        self.agents = self.possible_agents[:]

        # episode state
        self._step_count = 0
        self._targets: Optional[np.ndarray] = None
        self._prev_dists: Optional[np.ndarray] = None
        self.last_reward = np.zeros(self.num_drones, dtype=np.float32)

        # persistent reach memory (per episode):
        # once a drone reaches, it stays reached until reset
        self._reached_once: Optional[np.ndarray] = None

        # define the pybullet env
        self._env = VelocityAviary(
            drone_model=DroneModel.CF2X,
            num_drones=self.num_drones,
            physics=Physics.PYB,
            gui=(render_mode == "human"),
            record=False,
        )

        self._env.SPEED_LIMIT = 2.0 # m/s

        self._p = None
        self._client_id = None
        self._drone_ids: List[int] = []

        self._obstacle_ids: List[int] = []
        self._obstacle_base_pos: Optional[np.ndarray] = None
        self._obstacle_phase: Optional[np.ndarray] = None

        # GUI-only debug visuals
        self._target_body_ids: List[int] = []
        self._target_text_ids: List[int] = []
        self._lidar_line_ids: Dict[int, List[int]] = {}

        self._init_pybullet_handles()

        # Action: (vx_dir, vy_dir, vz_dir, speed)
        self._action_space = spaces.Box(
            low=np.full((4,), -1.0, dtype=np.float32),
            high=np.full((4,), 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        # base observations
        # pos(3), vel(3), to_target(3), to_nearest_obstacle(3), to_centroid(3), to_nearest_teammate(3) = 18
        self._base_obs_dim = 18

        # LiDAR observations
        self._lidar_dim = (LIDAR_NUM_RAYS * LIDAR_NUM_LAYERS) if self.lidar else 0

        # total observations
        obs_dim = self._base_obs_dim + self._lidar_dim
        self._obs_space = spaces.Box(
            low=np.full((obs_dim,), -1e9, dtype=np.float32),
            high=np.full((obs_dim,), 1e9, dtype=np.float32),
            dtype=np.float32,
        )

        if self.lidar:
            self._lidar_dirs_body = self._precompute_lidar_dirs_body()

        # csv logging
        self.log_path = Path("episode_stats.csv")

        # create file + header
        if not self.log_path.exists():
            with open(str(self.log_path), "w", newline = "") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode",
                    "steps",
                    "collisions",
                    "goals_reached",
                    "walls_hits",
                    "mean_reward"
                ])
        self.ep_rewards = []


    def _init_pybullet_handles(self) -> None:
        try:
            import pybullet as p

            self._p = p

            self._client_id = None
            for attr in ("_CLIENT", "CLIENT", "client", "_client"):
                if hasattr(self._env, attr):
                    self._client_id = getattr(self._env, attr)
                    break

            try:
                self._call_physics(p.setGravity, 0, 0, -9.8)
            except Exception:
                pass

            self._drone_ids = []
            for attr in ("DRONE_IDS", "drone_ids", "_drone_ids"):
                if hasattr(self._env, attr):
                    try:
                        self._drone_ids = list(getattr(self._env, attr))
                        break
                    except Exception:
                        pass

        except Exception:
            self._p = None
            self._client_id = None
            self._drone_ids = []


    def observation_space(self, agent):
        """ Retrieve the observation space """
        return self._obs_space


    def action_space(self, agent):
        """ Retrieve the action space """
        return self._action_space

    # ---- RESET FUNCTIONS ----
    def reset(self, *, seed=None, options=None):
        """ Reset the environment. Save episode stats in csv file for debugging. """
        if self._step_count > 0:
            self.write_ep_to_csv()

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._step_count = 0
        self.agents = self.possible_agents[:]

        self._env.reset()
        self._init_pybullet_handles()
        self._refresh_drone_ids_from_pybullet()
        # retrieve and copy the drone spawn positions
        self._spawn_positions = self._active_positions().copy()

        self.ep_done = False
        self.wall_hits = 0
        self.dead[:] = False
        self.collision_count = 0
        self.reached_total = 0
        self._reached_once = np.zeros((self.num_drones,), dtype=np.bool_)

        # reset environment
        self._spawn_obstacles()
        self._targets = self._make_targets()
        self._clear_debug_visuals()
        self._spawn_target_markers()

        # bookkeeping
        self._prev_dists = np.zeros((self.num_drones,), dtype=np.float32)
        self.last_reward = np.zeros(self.num_drones, dtype=np.float32)
        self._status = np.zeros(self.num_drones, dtype=np.int32)

        obs0 = self._build_obs_dict()
        for i in range(self.num_drones):
            if i >= self.active_drones:
                self._prev_dists[i] = 0.0
            else:
                pos = obs0[self.possible_agents[i]][0:3]
                tgt = self._targets[i]
                self._prev_dists[i] = float(np.linalg.norm(tgt - pos))

        info_dict = {a: {} for a in self.agents}

        return obs0, info_dict

    # ----- OBSERVATION VECTOR FUNCTIONS -----
    def _build_obs_dict(self) -> Dict[str, np.ndarray]:
        """Build a dictionary containing every drone's observation space vector"""

        obs_dict: Dict[str, np.ndarray] = {}

        # if drone does not yet have target, create one
        if self._targets is None:
            self._targets = self._make_targets()

        # compute drone positions and swarm centre
        positions = self._active_positions()
        centroid = self._swarm_centroid(positions)

        for i, a in enumerate(self.possible_agents):
            # if drone not active, give it observation zero and skip the rest
            if i >= self.active_drones:
                obs_dict[a] = np.zeros((self._base_obs_dim + self._lidar_dim,), dtype=np.float32)
                continue

            # get raw state vector from simulator, and extract position and velocity
            state = self._env._getDroneStateVector(i)
            pos = np.array(state[0:3], dtype=np.float32)
            vel = np.array(state[10:13], dtype=np.float32)

            # compute distance to assigned target
            target = self._targets[i]
            to_target = (target - pos).astype(np.float32)

            # compute distance to nearest obstacle, swarm centre and drone
            to_obs = self._nearest_obstacle_vector(pos)
            to_centroid = (centroid - pos).astype(np.float32)
            to_nearest_teammate = self._nearest_teammate_vector(i, positions)

            # form the base observation vector
            base = np.concatenate([pos, vel, to_target, to_obs, to_centroid, to_nearest_teammate],
                                  axis=0).astype(np.float32)

            # add lidar observation to base vector
            if self.lidar:
                scan = self._lidar_scan(i, pos)
                obs_dict[a] = np.concatenate([base, scan], axis=0).astype(np.float32)
            else:
                obs_dict[a] = base

        return obs_dict


    def _active_positions(self) -> np.ndarray:
        """ Retrieve all the active drone positions. """
        pos = np.zeros((self.active_drones, 3), dtype=np.float32)
        for i in range(self.active_drones):
            state = self._env._getDroneStateVector(i)
            pos[i] = np.array(state[0:3], dtype=np.float32)

        return pos


    def _swarm_centroid(self, positions: np.ndarray) -> np.ndarray:
        """ Compute the swarm centroid. """
        if positions.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)

        return positions.mean(axis=0).astype(np.float32)


    def _nearest_teammate_vector(self, i: int, positions: np.ndarray) -> np.ndarray:
        """ Retrieve the vector to the nearest teammate."""
        if positions.shape[0] <= 1:
            return np.zeros(3, dtype=np.float32)
        pi = positions[i]
        d = positions - pi[None, :]
        dist = np.linalg.norm(d, axis=1) + 1e-8
        dist[i] = 1e9
        j = int(np.argmin(dist))

        return (positions[j] - pi).astype(np.float32)


    def write_ep_to_csv(self):
        """ Write episode stats to csv """

        mean_reward = float(np.mean(self.ep_rewards)) if self.ep_rewards else 0.0

        with open(self.log_path, "a", newline = "") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.ep_idx,
                self._step_count,
                self.collision_count,
                self.reached_total,
                self.wall_hits,
                mean_reward
            ])
        self.ep_idx += 1
        self.ep_rewards.clear()


    def is_far_from_drones(self, point: np.ndarray, min_dist:float) -> bool:
        """ Determine if potential obstacle spawn positions is far from the drone spawn positions."""
        if self._spawn_positions is None:
            return True
        d = np.linalg.norm(self._spawn_positions - point, axis=1)

        return np.all(d >= min_dist)


    # ----- PYBULLET CLIENT FUNCTIONS -----
    def _connected_ids(self) -> List[int]:
        """ Scan and return a list of all active PyBullet Client IDs (simulation connections) """
        if self._p is None:
            return []
        p = self._p
        ids: List[int] = []
        for i in range(64):
            try:
                if p.isConnected(i):
                    ids.append(int(i))
            except Exception:
                continue
        return ids


    def _pcid_for_physics(self) -> Optional[int]:
        """ Pick correct client to run physics in """
        if self._p is None:
            return None
        p = self._p
        if self._client_id is not None:
            try:
                if p.isConnected(self._client_id):
                    return int(self._client_id)
            except Exception:
                pass
        ids = self._connected_ids()
        return ids[0] if ids else None


    def _pcid_for_gui_draw(self) -> Optional[int]:
        """ Pick correct client to run gui in """
        ids = self._connected_ids()
        return max(ids) if ids else None


    def _call_physics(self, fn, *args, **kwargs):
        """ Find physics client, add label and call function. """
        if self._p is None:
            raise RuntimeError("PyBullet not available")

        # retrieve id of physics simulation
        pcid = self._pcid_for_physics()
        if pcid is not None and "physicsClientId" not in kwargs:
            try:
                return fn(*args, **kwargs, physicsClientId=pcid)
            except Exception:
                pass
        return fn(*args, **kwargs)


    def _call_gui(self, fn, *args, **kwargs):
        """ Find gui client, add label and call function. """
        if self._p is None:
            return None

        # retrieve id of gui simulation
        pcid = self._pcid_for_gui_draw()
        try:
            if pcid is not None and "physicsClientId" not in kwargs:
                return fn(*args, **kwargs, physicsClientId=pcid)
            return fn(*args, **kwargs)
        except Exception:
            return None


    def _refresh_drone_ids_from_pybullet(self) -> None:
        """ Check PyBullet simulation itself to find the drones - useful if env does not give it. """
        if self._p is None:
            return
        p = self._p

        try:
            num_bodies = self._call_physics(p.getNumBodies)
        except Exception:
            return

        ids: List[int] = []
        for i in range(num_bodies):
            try:
                body_id = self._call_physics(p.getBodyUniqueId, i)
                info = self._call_physics(p.getBodyInfo, body_id)
                name = info[1].decode("utf-8", errors="ignore").lower()
                if ("crazyflie" in name) or (name.startswith("cf")) or ("drone" in name):
                    ids.append(int(body_id))
            except Exception:
                continue

        if len(ids) == 0:
            ids = []
            for i in range(num_bodies):
                try:
                    ids.append(int(self._call_physics(p.getBodyUniqueId, i)))
                except Exception:
                    pass

        self._drone_ids = ids

    # ------ TARGET FUNCTIONS ------
    def make_valid_target(self, min_dist = 0.5, max_attempts = 1000) -> np.ndarray:
        """
        Make a valid target for all drones, i.e. a target that does not collide with the drone itself, other obstacles
        or other drones.
        """
        for _ in range(max_attempts):
            t = np. array([
                self.rng.uniform(-self.world_scale, self.world_scale),
                self.rng.uniform(-self.world_scale, self.world_scale),
                self.rng.uniform(0.8, 2.0)
            ], dtype=np.float32)

            drone_positions = self._spawn_positions
            if np.any(np.linalg.norm(drone_positions - t, axis=1) < min_dist):
                continue

            obs_pos = []
            for oid in self._obstacle_ids:
                try:
                    pos, _ = self._call_physics(self._p.getBasePositionAndOrientation, oid)
                    obs_pos.append(pos)
                except Exception:
                    pass

            obs_pos = np.array(obs_pos, dtype=np.float32)
            if len(obs_pos) > 0:
                if np.any(np.linalg.norm(obs_pos - t, axis=-1) < (OBSTACLE_RADIUS + min_dist)):
                    continue

            return t

        x = self.rng.uniform(-self.world_scale, self.world_scale)
        y = self.rng.uniform(-self.world_scale, self.world_scale)
        z = self.rng.uniform(0.8, 2.0)

        return np.array([x, y, z], dtype=np.float32)


    def _make_targets(self) -> np.ndarray:
        """ Make a target for all drones."""
        targets = []

        if DEFAULT_SHARED_TARGET:
            t = self.make_valid_target()

            # repeat shared target for all drones
            return np.repeat(t[None, :], repeats=self.num_drones, axis=0).astype(np.float32)

        for _ in range(self.num_drones):
            t = self.make_valid_target()
            targets.append(t)

        return np.array(targets, dtype=np.float32)


    def _spawn_target_markers(self) -> None:
        """ Spawn the target markers on the GUI. """
        if self._p is None or self._targets is None or self.render_mode != "human":
            return
        p = self._p

        vis = self._call_gui(
            p.createVisualShape,
            p.GEOM_SPHERE,
            radius=0.10,
            rgbaColor=[0.1, 1.0, 0.1, 0.9],
        )
        if vis is None:
            return

        tgt = self._targets[0] if DEFAULT_SHARED_TARGET else None

        if DEFAULT_SHARED_TARGET and tgt is not None:
            body = self._call_gui(
                p.createMultiBody,
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vis,
                basePosition=[float(tgt[0]), float(tgt[1]), float(tgt[2])],
            )
            if body is not None:
                self._target_body_ids.append(int(body))

            tid = self._call_gui(
                p.addUserDebugText,
                "T",
                [float(tgt[0]), float(tgt[1]), float(tgt[2] + 0.15)],
                textColorRGB=[0.2, 1.0, 0.2],
                textSize=1.4,
                lifeTime=0,
            )
            if tid is not None:
                self._target_text_ids.append(int(tid))
            return

        for i in range(self.num_drones):
            tgt_i = self._targets[i]
            body = self._call_gui(
                p.createMultiBody,
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vis,
                basePosition=[float(tgt_i[0]), float(tgt_i[1]), float(tgt_i[2])],
            )
            if body is not None:
                self._target_body_ids.append(int(body))

            tid = self._call_gui(
                p.addUserDebugText,
                f"T{i}",
                [float(tgt_i[0]), float(tgt_i[1]), float(tgt_i[2] + 0.15)],
                textColorRGB=[0.2, 1.0, 0.2],
                textSize=1.2,
                lifeTime=0,
            )
            if tid is not None:
                self._target_text_ids.append(int(tid))


    # ------ OBSTACLE FUNCTIONS --------
    def _spawn_obstacles(self) -> None:
        """ Spawn the obstacles. """

        self._obstacle_ids = []
        self._obstacle_base_pos = None
        self._obstacle_phase = None

        if self.num_obstacles <= 0 or self._p is None:
            return

        p = self._p
        col = self._call_physics(p.createCollisionShape, p.GEOM_SPHERE, radius=OBSTACLE_RADIUS)
        vis = self._call_physics(p.createVisualShape, p.GEOM_SPHERE, radius=OBSTACLE_RADIUS,
                                 rgbaColor=[1, 0, 0, 1])

        base_positions = []
        phases = []
        candidate = []

        for _ in range(self.num_obstacles):

            for i in range(1000):

                x = self.rng.uniform(-self.world_scale, self.world_scale)
                y = self.rng.uniform(-self.world_scale, self.world_scale)
                z = self.rng.uniform(OBSTACLE_HEIGHT_RANGE[0], OBSTACLE_HEIGHT_RANGE[1])

                candidate = np.array([x,y,z], dtype=np.float32)

                if not self.is_far_from_drones(candidate, OBSTACLE_RADIUS + 0.4): # TODO ASK PEDRO ABOUT 0.4
                    continue

                break

            body = self._call_physics(
                p.createMultiBody,
                baseMass=0.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=candidate.tolist(),
            )
            self._obstacle_ids.append(int(body))
            base_positions.append(candidate.tolist())
            phases.append(self.rng.uniform(0.0, 2.0 * np.pi))

        self._obstacle_base_pos = np.array(base_positions, dtype=np.float32)
        self._obstacle_phase = np.array(phases, dtype=np.float32)


    def _update_obstacles(self) -> None:
        """ Update the dynamic obstacles. """
        if (
                (not OBSTACLE_DYNAMIC)
                or self._p is None
                or self._obstacle_base_pos is None
                or self._obstacle_phase is None
                or len(self._obstacle_ids) == 0
        ):
            return

        p = self._p
        t = float(self._step_count) * (1.0 / 60.0)

        for k, oid in enumerate(self._obstacle_ids):
            base = self._obstacle_base_pos[k]
            ph = float(self._obstacle_phase[k])

            dx = OBSTACLE_DRIFT_AMP * np.cos(OBSTACLE_DRIFT_OMEGA * t + ph)
            dy = OBSTACLE_DRIFT_AMP * np.sin(OBSTACLE_DRIFT_OMEGA * t + ph)

            pos = [float(base[0] + dx), float(base[1] + dy), float(base[2])]
            self._call_physics(p.resetBasePositionAndOrientation, oid, posObj=pos,
                               ornObj=[0, 0, 0, 1])


    def _nearest_obstacle_vector(self, drone_pos: np.ndarray) -> np.ndarray:
        if self._p is None or len(self._obstacle_ids) == 0:
            return np.zeros(3, dtype=np.float32)

        p = self._p
        obs_pos = []
        for oid in self._obstacle_ids:
            try:
                pos, _ = self._call_physics(p.getBasePositionAndOrientation, oid)
                obs_pos.append(pos)
            except Exception:
                pass

        if len(obs_pos) == 0:
            return np.zeros(3, dtype=np.float32)

        obs_pos = np.array(obs_pos, dtype=np.float32)
        d = obs_pos - drone_pos[None, :]
        dist = np.linalg.norm(d, axis=1) + 1e-8
        idx = int(np.argmin(dist))
        return d[idx].astype(np.float32)


    # def _check_collisions(self) -> bool:
    #     """Collision check against obstacles and other drones. """
    #     if self._p is None or len(self._drone_ids) == 0:
    #         return False
    #
    #     p = self._p
    #
    #     thresh_obstacle = OBSTACLE_RADIUS + COLLISION_DIST
    #     thresh_drone = COLLISION_DIST
    #     active_ids = self._drone_ids[: self.active_drones] if len(
    #         self._drone_ids) >= self.active_drones else self._drone_ids
    #
    #     # drone vs obstacle
    #     if len(self._obstacle_ids) >0:
    #         for did in active_ids:
    #             for oid in self._obstacle_ids:
    #                 try:
    #                     pts = self._call_physics(p.getClosestPoints, bodyA=did, bodyB=oid,
    #                                              distance=thresh_obstacle)
    #                     if pts and len(pts) > 0:
    #                         return True
    #                 except Exception:
    #                     continue
    #
    #     # drone vs drone
    #     for i, did1 in enumerate(active_ids):
    #         for did2 in active_ids[i+1:]:
    #             try:
    #                 pts = self._call_physics(p.getClosestPoints, bodyA=did1, bodyB=did2,
    #                                          distance=thresh_drone)
    #                 if pts and len(pts) > 0:
    #                     return True
    #             except Exception:
    #                 continue
    #
    #     return False


    def check_collisions_per_drone(self):
        """
        Returns:
            collided[i] is True if drone i collided with anything

        """
        collided = np.zeros(self.num_drones, dtype=np.bool_)

        if self._p is None or len(self._drone_ids) == 0:
            return collided

        p = self._p
        active_ids = self._drone_ids[: self.active_drones]

        thresh_obstacle = OBSTACLE_RADIUS + COLLISION_DIST
        thresh_drone = COLLISION_DIST

        # drone vs obstacle
        if len(self._obstacle_ids) >0:
            for i, did in enumerate(active_ids):
                for oid in self._obstacle_ids:
                    try:
                        pts = self._call_physics(p.getClosestPoints, bodyA=did, bodyB=oid,
                                                 distance=thresh_obstacle)
                        if pts and len(pts) > 0:
                            collided[i] = True
                            break
                    except Exception:
                        continue

        # drone vs drone
        for i, did1 in enumerate(active_ids):
            for j, did2 in enumerate(active_ids[i+1:], start = i+1):
                try:
                    pts = self._call_physics(p.getClosestPoints, bodyA=did1, bodyB=did2,
                                             distance=thresh_drone)
                    if pts and len(pts) > 0:
                        collided[i] = True
                        collided[j] = True
                except Exception:
                    continue

        return collided


    # -------- GUI-ONLY VISUALS --------
    def _clear_debug_visuals(self) -> None:
        if self._p is None or self.render_mode != "human":
            self._target_body_ids = []
            self._target_text_ids = []
            self._lidar_line_ids = {}
            return

        p = self._p

        for bid in self._target_body_ids:
            self._call_gui(p.removeBody, bid)
        self._target_body_ids = []

        for tid in self._target_text_ids:
            self._call_gui(p.removeUserDebugItem, tid)
        self._target_text_ids = []

        for _, line_ids in list(self._lidar_line_ids.items()):
            for lid in line_ids:
                self._call_gui(p.removeUserDebugItem, lid)
        self._lidar_line_ids = {}


    # ------ LIDAR METHODS ------
    def _precompute_lidar_dirs_body(self) -> np.ndarray:

        h_fov = np.deg2rad(max(1e-6, LIDAR_HORIZONTAL_FOV_DEG))
        v_fov = np.deg2rad(max(0.0, LIDAR_VERTICAL_FOV_DEG))

        if LIDAR_HORIZONTAL_FOV_DEG >= 359.0:
            az = np.linspace(0.0, 2.0 * np.pi, num=LIDAR_NUM_RAYS, endpoint=False,
                             dtype=np.float32)
        else:
            az = np.linspace(-0.5 * h_fov, 0.5 * h_fov, num=LIDAR_NUM_RAYS, endpoint=True,
                             dtype=np.float32)

        if LIDAR_NUM_LAYERS <= 1 or LIDAR_VERTICAL_FOV_DEG <= 1e-6:
            el = np.array([0.0], dtype=np.float32)
        else:
            el = np.linspace(-0.5 * v_fov, 0.5 * v_fov, num=LIDAR_NUM_LAYERS, endpoint=True,
                             dtype=np.float32)

        dirs = []
        for e in el:
            ce = float(np.cos(e))
            se = float(np.sin(e))
            for a in az:
                ca = float(np.cos(a))
                sa = float(np.sin(a))
                dirs.append([ce * ca, ce * sa, se])

        dirs = np.array(dirs, dtype=np.float32)
        n = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8

        return (dirs / n).astype(np.float32)


    def _rotmat_from_quat(self, qxyzw: np.ndarray) -> np.ndarray:

        x, y, z, w = [float(v) for v in qxyzw]
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float32,
        )


    def _lidar_scan(self, drone_idx: int, pos: np.ndarray) -> np.ndarray:

        if (not self.lidar) or (self._p is None) or (self._lidar_dim <= 0):
            return np.zeros((0,), dtype=np.float32)

        p = self._p
        R = np.eye(3, dtype=np.float32)

        did = self._drone_ids[drone_idx] if (0 <= drone_idx < len(self._drone_ids)) else None
        if did is not None:
            try:
                _, orn = self._call_physics(p.getBasePositionAndOrientation, did)
                R = self._rotmat_from_quat(np.array(orn, dtype=np.float32))
            except Exception:
                pass

        origin = pos.astype(np.float32).copy()
        origin[2] += 0.03

        dirs_world = (R @ self._lidar_dirs_body.T).T
        ray_from = np.repeat(origin[None, :], dirs_world.shape[0], axis=0)
        ray_to = ray_from + dirs_world * LIDAR_RANGE

        try:
            results = self._call_physics(
                p.rayTestBatch,
                rayFromPositions=ray_from.tolist(),
                rayToPositions=ray_to.tolist(),
            )
        except Exception:
            return np.ones((dirs_world.shape[0],), dtype=np.float32)

        hit_frac = np.ones((dirs_world.shape[0],), dtype=np.float32)
        hit_pos = ray_to.copy()

        for i, r in enumerate(results):
            try:
                hf = float(r[2])
                hit_frac[i] = np.clip(hf, 0.0, 1.0)
                hp = r[3]
                hit_pos[i] = np.array(hp, dtype=np.float32)
            except Exception:
                pass

        if LIDAR_NOISE_STD > 0.0:
            hit_frac = np.clip(
                hit_frac + self.rng.normal(0.0, LIDAR_NOISE_STD, size=hit_frac.shape).astype(
                    np.float32),
                0.0,
                1.0,
            )

        # GUI debug drawing
        if self.render_mode == "human" and LIDAR_SHOW_RAYS_TEST:
            old = self._lidar_line_ids.get(drone_idx, [])
            for lid in old:
                self._call_gui(p.removeUserDebugItem, lid)

            new_ids: List[int] = []
            for i in range(dirs_world.shape[0]):
                endp = hit_pos[i] if hit_frac[i] < 0.999 else ray_to[i]
                col = [1.0, 0.2, 0.2] if hit_frac[i] < 0.999 else [0.2, 0.6, 1.0]
                lid = self._call_gui(
                    p.addUserDebugLine,
                    lineFromXYZ=ray_from[i].tolist(),
                    lineToXYZ=endp.tolist(),
                    lineColorRGB=col,
                    lineWidth=1.0,
                    lifeTime=0,
                )
                if lid is not None:
                    new_ids.append(int(lid))
            self._lidar_line_ids[drone_idx] = new_ids

        return hit_frac.astype(np.float32)


    # ----- STEP FUNCTION ------
    def step(self, actions: Dict[str, np.ndarray]):
        """
        One environment step.

        Flow:
        1. Update world (moving obstacles, counters, etc.)
        2. Freeze drones that already reached the goal.
        3. Convert policy actions into simulator velocity commands.
        4. Step the physics.
        5. Build observations.
        6. Compute the swarm stats and termination flags (global).
        7. Compute per-agent rewards.
        8. Build the info dict.

        Args:
            actions: ndarray, shape (n, 4);
                    (vx_dir, vy_dir, vz_dir, speed)

        Returns:
            obs_dict, reward_dict, term_dict, trunc_dict, info_dict
        """

        # 1. Update the world
        self._step_count += 1
        self._update_obstacles()

        # memory of who already reached the goal
        if self._reached_once is None:
            self._reached_once = np.zeros((self.num_drones,), dtype=np.bool_)
        # ---------------------------------------------------------------------
        # 2. Freeze drones that already reached the goal:
        # - IMPORTANT: to make speed=0 in the mapping, we must set u[3] = -1
        #   because u[3] in [-1,1] maps to speed01 in [0,1] by:
        #     speed01 = (u[3] + 1) * 0.5
        # ---------------------------------------------------------------------
        for i, a in enumerate(self.possible_agents):
            if i < self.active_drones and self._reached_once[i] and a in actions:
                actions[a] = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)

        # ---------------------------------------------------------------------
        # 3. Convert normalised policy actions into simulator (VelocityAviary) velocity commands
        # Policy output = [dir_x, dir_y, dir_z, speed] in [-1, 1]
        # Simulator expects: direction (unit vector) and speed (m/s)
        # ---------------------------------------------------------------------
        act = np.zeros((self.num_drones, 4), dtype=np.float32)

        for i, a in enumerate(self.possible_agents):

            # inactive drones -> just hover
            if i >= self.active_drones:
                act[i] = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
                continue

            # default is direction = up and speed = 0 m/s
            dir_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            speed = 0.0

            if a in actions:
                u = np.array(actions[a], dtype=np.float32).reshape(-1)
                # clip to valid range
                u = np.clip(u[:4], -1.0, 1.0)

                # get the required direction based on normalised action
                v = u[:3].astype(np.float32)
                n = np.linalg.norm(v)
                dir_vec = v/n if n > 1e-6 else dir_vec

                # speed mapping: u[3] from [-1,1] -> [0,1 (vel_scale)]
                speed = float((u[3] + 1.0) * 0.5)
                # speed = VEL_SCALE * speed01  # fraction, NOT m/s

                # hover bias to fight gravity drift
                dir_vec[2] += HOVER_BIAS_VZ
                dir_vec /= max(np.linalg.norm(dir_vec), 1e-6)

            # automatic takeoff phase (override RL for initial steps during training and tell drone to go up with takeoff velocity (off ground))
            if self._step_count <= TAKEOFF_STEPS:
                dir_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                speed = float(abs(TAKEOFF_VZ))

            # retrieve direction vector and speed for drone
            act[i, 0:3] = dir_vec
            act[i, 3] = float(speed)

        # 4. Step the simulator using the action
        self._env.step(act)

        # 5. Build the observation dictionary
        obs_dict = self._build_obs_dict()

        # 6. Global episode state (collision, swarm cohesion, time limit)
        time_limit = (self._step_count >= self.max_steps)

        collisions = self.check_collisions_per_drone()
        all_collided = np.all(collisions[:self.active_drones])

        positions = self._active_positions()
        max_sep, min_sep, too_far, too_close = self._pairwise_sep_stats(positions)

        # Swarm constraints
        swarm_term = (too_far and DEFAULT_TERMINATE_ON_COHESION_BREAK) or (
                too_close and DEFAULT_TERMINATE_ON_SEPARATION)

        all_active_reached = all(bool(self._reached_once[i]) for i in range(self.active_drones))
        episode_done = bool(all_collided or all_active_reached or swarm_term) # episode ends if everyone crashed, all active reached or spread too far

        self.dead = np.logical_or(self.dead, collisions)

        # Swarm shaping penalties
        cohesion_pen = 0.0
        if positions.shape[0] > 1 and max_sep > DEFAULT_SWARM_MAX_SEP:
            cohesion_pen = - COHESION_PEN * (max_sep - DEFAULT_SWARM_MAX_SEP)

        sep_pen = 0.0
        if positions.shape[0] > 1 and min_sep < DEFAULT_SWARM_MIN_SEP:
            sep_pen = -SEPARATION_PEN * (DEFAULT_SWARM_MIN_SEP - min_sep)

        # Track distances to goal for progress-reward
        if self._prev_dists is None:
            self._prev_dists = np.zeros((self.num_drones,), dtype=np.float32)

        # detect newly-reached goals for one-time bonuses in reward
        was_reached = self._reached_once.copy()
        reached_now = np.zeros((self.num_drones,), dtype=np.bool_)

        for i, a in enumerate(self.possible_agents):
            if i >= self.active_drones:
                continue

            pos = obs_dict[a][0:3]
            target = self._targets[i] if self._targets is not None else np.zeros(3,
                                                                                 dtype=np.float32)
            dist = float(np.linalg.norm(target - pos))

            if dist < GOAL_RADIUS:
                reached_now[i] = True
                self._reached_once[i] = True

            # allow a crashed drone to stop moving
            if self.dead[i] and a in actions:
                actions[a] = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)

        just_reached = np.logical_and(reached_now, np.logical_not(was_reached))

        # 7. Compute rewards (per agent)
        reward_dict = {}
        term_dict = {}
        trunc_dict = {}

        for i, a in enumerate(self.possible_agents):

            if i >= self.active_drones:
                reward_dict[a] = 0.0
                trunc_dict[a] = time_limit
                term_dict[a] = episode_done
                continue

            pos = obs_dict[a][0:3]
            target = self._targets[i] if self._targets is not None else np.zeros(3,
                                                                                 dtype=np.float32)

            dist = float(np.linalg.norm(target - pos))
            prev = float(self._prev_dists[i])
            self._prev_dists[i] = dist

            # reward terms
            r_progress = PROG_COEF * (prev - dist)
            r_goal_dist = -0.15 * dist
            r_time = -TIME_PEN
            r_ctrl = -CNTRL_PEN * float(np.linalg.norm(act[i]))
            if len(self._obstacle_ids) > 0:
                d_obs = float(np.linalg.norm(obs_dict[a][9:12]))
                r_near = -NEAR_OBS_PEN / max(d_obs, 0.10)
            else:
                r_near = 0.0

            r = (
                    r_progress +
                    r_goal_dist +
                    r_time +
                    r_ctrl +
                    r_near +
                    cohesion_pen +
                    sep_pen
            )

            # goal bonus (occurs only once): when it first becomes reached
            if just_reached[i]:
                r += GOAL_BONUS

            if collisions[i]:
                r -= COLLISION_PEN # full penalty to culprit
            elif np.any(collisions):
                r -= 0.2 * COLLISION_PEN # small team penalty if one drone crashes

            reward_dict[a] = float(r)
            term_dict[a] = bool(self.dead[i] or episode_done) # allow only crashed drone to terminate
            trunc_dict[a] = bool(time_limit)

        # keep fixed agent list
        self.agents = self.possible_agents[:]

        #  8. Info dict (for callbacks & logging)
        info_dict = {}

        for i, a in enumerate(self.possible_agents):

            if i >= self.active_drones:
                info_dict[a] = {
                    "step": int(self._step_count),
                    "active": False,
                    "collided": bool(collisions[i]),
                    "reached": False,
                    "swarm_max_sep": float(max_sep),
                    "swarm_min_sep": float(min_sep),
                    "swarm_too_far": bool(too_far),
                    "swarm_too_close": bool(too_close),
                    "episode_done_collision": bool(collisions[i]),
                    "episode_done_all_reached": bool(all_active_reached),
                    "episode_done_swarm": bool(swarm_term),
                }
                continue

            info_dict[a] = {
                "step": int(self._step_count),
                "active": True,
                "collided": bool(collisions[i]),
                "reached": bool(self._reached_once[i]),
                "reached_now": bool(reached_now[i]),
                "just_reached": bool(just_reached[i]),
                "swarm_max_sep": float(max_sep),
                "swarm_min_sep": float(min_sep),
                "swarm_too_far": bool(too_far),
                "swarm_too_close": bool(too_close),
                "episode_done_collision": bool(collisions[i]),
                "episode_done_all_reached": bool(all_active_reached),
                "episode_done_swarm": bool(swarm_term),
            }

        self.agents = self.possible_agents[:]

        step_mean_reward = float(np.mean([reward_dict[a] for a in self.possible_agents[:self.active_drones]]))
        self.ep_rewards.append(step_mean_reward)

        self.collision_count += int(np.sum(collisions[:self.active_drones]))

        # count “first time reach”
        self.reached_total += int(np.sum(just_reached[:self.active_drones]))

        return obs_dict, reward_dict, term_dict, trunc_dict, info_dict


    def _pairwise_sep_stats(self, positions: np.ndarray) -> Tuple[float, float, bool, bool]:
        """
        Check every unique drone pair and retrieve swarm drone stats.
        Computes:
          max_d   = maximum pairwise distance
          min_d   = minimum pairwise distance
          too_far = any pair exceeds swarm_max_sep
          too_close = any pair below swarm_min_sep
        """
        n = positions.shape[0]
        if n <= 1:
            return 0.0, 1e9, False, False

        max_d = 0.0
        min_d = 1e9
        too_far = False
        too_close = False

        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(positions[i] - positions[j]))
                max_d = max(max_d, d)
                min_d = min(min_d, d)
                if d > DEFAULT_SWARM_MAX_SEP:
                    too_far = True
                if d < DEFAULT_SWARM_MIN_SEP:
                    too_close = True

        return max_d, min_d, too_far, too_close

    def render(self):
        return None

    def close(self):
        try:
            self._clear_debug_visuals()
        except Exception:
            pass
        try:
            self._env.close()
        except Exception:
            pass


def env(
        num_drones=2,
        active_drones=None,
        render_mode="human",
        num_obstacles=4,
        max_steps=500,
        lidar=False,
        **kwargs,
):
    return DroneSwarm3D(
        num_drones=num_drones,
        active_drones=active_drones,
        num_obstacles=num_obstacles,
        lidar=lidar,
        max_steps=max_steps,
        render_mode=render_mode,
        **kwargs,
    )
