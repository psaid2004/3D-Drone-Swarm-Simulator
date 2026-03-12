from __future__ import annotations

import csv
from typing import Dict, Optional, Tuple, List
from pathlib import Path

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from envs.createObjects import Target, EnvObject, Dynamic
from wrappers.step_logic import (
    freeze_inactive_actions,
    convert_actions_for_simulator,
    compute_step_state,
    compute_terminations,
    build_info_dict,
)
from wrappers.rewards import compute_rewards
from wrappers.stats_trackers import EpisodeStatsTracker

from config.defaults import *

#TODO despawn after reaching target, spawn obstacle between target and drone?

class DroneSwarm3D(ParallelEnv):

    metadata = {"name": "drone_swarms_3d", "render_mode": "human"}

    # ----------------------------- Initialise Custom 3D Environment -----------------------------

    def __init__(
            self,
            world_scale: float,
            num_drones_per_swarm: int,
            num_static_obstacles: int,
            num_dynamic_obstacles: int,
            max_steps: int,
            num_swarms: int = 1,
            lidar: bool = False,
            render_mode: Optional[str] = None,
            seed: Optional[int] = None
    ):

        self._init_config(
            world_scale=world_scale,
            num_drones_per_swarm=num_drones_per_swarm,
            num_static_obstacles=num_static_obstacles,
            num_dynamic_obstacles=num_dynamic_obstacles,
            max_steps=max_steps,
            num_swarms=num_swarms,
            lidar=lidar,
            render_mode=render_mode,
        )

        self._init_rng(seed)
        self._init_agents()
        self._init_episode_state()
        self._init_simulator()
        self._init_spaces()
        self._init_logging()


    def _init_config(
            self,
            world_scale: float,
            num_drones_per_swarm: int,
            num_static_obstacles: int,
            num_dynamic_obstacles: int,
            max_steps: int,
            num_swarms: int,
            lidar: bool,
            render_mode: Optional[str],
    ):
        """ Store environment configuration. """
        self.world_scale = float(world_scale)
        self.num_swarms = int(num_swarms)
        self.num_drones_per_swarm = int(num_drones_per_swarm)

        # ensure at least one drone in environment
        self.total_num_drones = max(
            1,
            self.num_drones_per_swarm * self.num_swarms
        ) # TODO FUTURE IMPLEMENTATION SHOULD CONSIDER NUM SWARMS - ASSUMPTION OF 1 SWARM HERE

        # Spawn position of every drone
        self.spawn_positions = np.empty((0,3))

        # Dict with swarm: [position of drone]
        self.drones_in_swarm ={}

        self.num_static_obstacles = int(num_static_obstacles)
        self.num_dynamic_obstacles = int(num_dynamic_obstacles)
        self.max_steps = int(max_steps)
        self.lidar = bool(lidar)
        self.render_mode = render_mode


    def _init_rng(self, seed: Optional[int]) -> None:
        """ Create per-environment random number generator. """
        self.rng = np.random.default_rng(seed)


    def _init_agents(self) -> None:
        """ Initialize PettingZoo agents. """
        self.possible_agents = [f"drone_{i}" for i in
                                range(self.total_num_drones)]  # fixed list of all drone names in env
        self.agents = self.possible_agents[:]  # current active list in episode (used to control dead / despawned drones)


    def _init_episode_state(self) -> None:
        """ Initialise per-episode bookkeeping. """
        # boolean array representing death status of each drone
        self._dead = np.zeros(self.total_num_drones, dtype=np.bool_)

        # boolean array of target reached memory: once reached, stays True until reset
        self._reached_once = np.zeros((self.total_num_drones,), dtype=np.bool_)
        self._despawned = np.zeros((self.total_num_drones,), dtype=np.bool_)
        self._spawn_positions = np.zeros((self.total_num_drones, 3), dtype=np.float32)

        # target for each drone; self._targets[i] belongs to drone i
        self._targets: Optional[list[Target]] = None

        # distance of each drone to its target in previous step (used for reward computation)
        self._prev_dists = np.zeros(self.total_num_drones, dtype=np.float32)
        # reward value of each drone in previous step
        self._last_reward = np.zeros(self.total_num_drones, dtype=np.float32)

        self._wall_collision_once = np.zeros((self.total_num_drones,), dtype=np.bool_)
        self._obstacle_collision_once = np.zeros((self.total_num_drones,), dtype=np.bool_)
        self._drone_collision_once = np.zeros((self.total_num_drones,), dtype=np.bool_)

        self._step_count = 0


    def _init_simulator(self) -> None:
        """ Create drone simulator and PyBullet handles. """
        # list of drone's initial position

        for i in range(self.num_swarms):
            center, drones = self.generate_drone_swarm_spawn()
            self.drones_in_swarm[i] = {"center": center, "drones":drones}
            self.spawn_positions = np.vstack([self.spawn_positions, drones])

        # --- Pybullet env ---
        self._env = VelocityAviary(
            drone_model=DroneModel.CF2X,
            num_drones=self.total_num_drones,
            initial_xyzs=self.spawn_positions,
            physics=Physics.PYB,
            gui=(self.render_mode == "human"),
            record=False,
            user_debug_gui=False,
        )

        self._env.SPEED_LIMIT = 2.0 # m/s

        self._p = None
        self._client_id = None
        self._drone_ids: List[int] = []
        self._obstacles = []

        # GUI-only debug visuals
        self._lidar_line_ids: Dict[int, List[int]] = {}

        self._init_pybullet_handles()
        self._refresh_drone_ids_from_pybullet()


    def _init_spaces(self) -> None:
        """ Create action and observation spaces. """

        # Action: (vx_dir, vy_dir, vz_dir, speed)
        self._action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape = (4,),
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


    def _init_logging(self) -> None:
        """ Create CSV log file if missing. """

        self.stats = EpisodeStatsTracker("episode_stats.csv")


    def _init_pybullet_handles(self) -> None:
        try:
            import pybullet as p

            self._p = p

            for attr in ("_CLIENT", "CLIENT", "client", "_client"):
                if hasattr(self._env, attr):
                    self._client_id = getattr(self._env, attr)
                    break

            try:
                self._call_physics(p.setGravity, 0, 0, -9.8)
            except Exception:
                pass

            for attr in ("DRONE_IDS", "drone_ids", "_drone_ids"):
                if hasattr(self._env, attr):
                    try:
                        self._drone_ids = list(getattr(self._env, attr))
                        break
                    except Exception:
                        pass

        except Exception:
            pass


    # ----------------------------- PYBULLET CLIENT FUNCTIONS -----------------------------


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

    def _pybullet_client_alive(self) -> bool:
        """Return True if the VelocityAviary's physics client is still connected."""
        try:
            import pybullet as p
        except Exception:
            return False

        cid = getattr(self._env, "CLIENT", None)
        if cid is None:
            cid = getattr(self._env, "_CLIENT", None)

        if cid is None:
            return False

        try:
            return bool(p.isConnected(cid))
        except Exception:
            return False

    def _rebuild_aviary(self) -> None:
        """Recreate VelocityAviary from scratch (fresh PyBullet connection)."""
        try:
            # best effort close old env (may already be half-dead)
            self._env.close()
        except Exception:
            pass

        # TODO Ask stella about neighbourhood radius
        self._env = VelocityAviary(
            drone_model=DroneModel.CF2X,
            num_drones=self.total_num_drones,
            initial_xyzs=self.spawn_positions,
            physics=Physics.PYB,
            gui=(self.render_mode == "human"),
            record=False,
            user_debug_gui=False,
        )
        self._env.SPEED_LIMIT = 2.0

        # reset objects tied to the old simulation
        self._obstacles = []
        self._lidar_line_ids = {}

        # reobtain pybullet handles and drone IDs
        self._init_pybullet_handles()


    # ----------------------------- (DE)SPAWN DRONES, OBSTACLES AND TARGETS -----------------------------


    def generate_drone_swarm_spawn(self):
        max_attempts = 1000

        for i in range(max_attempts):
            edge = self.rng.integers(0, 4)

            if edge == 0:  # +X edge
                center = [self.world_scale - SWARM_EDGE_MARGIN,
                    self.rng.uniform(-self.world_scale + SWARM_EDGE_MARGIN, self.world_scale -SWARM_EDGE_MARGIN),
                    self.rng.uniform(SWARM_EDGE_MARGIN, self.world_scale - SWARM_EDGE_MARGIN)
                ]

            elif edge == 1:  # -X edge
                center = [
                    -self.world_scale + SWARM_EDGE_MARGIN,
                    self.rng.uniform(-self.world_scale+SWARM_EDGE_MARGIN, self.world_scale-SWARM_EDGE_MARGIN),
                    self.rng.uniform(SWARM_EDGE_MARGIN, self.world_scale - SWARM_EDGE_MARGIN)
                ]

            elif edge == 2:  # +Y edge
                center = [
                    self.rng.uniform(-self.world_scale+SWARM_EDGE_MARGIN, self.world_scale-SWARM_EDGE_MARGIN),
                    self.world_scale - SWARM_EDGE_MARGIN,
                    self.rng.uniform(SWARM_EDGE_MARGIN, self.world_scale - SWARM_EDGE_MARGIN)
                ]

            else:  # -Y edge
                center = [
                    self.rng.uniform(-self.world_scale+SWARM_EDGE_MARGIN, self.world_scale-SWARM_EDGE_MARGIN),
                    -self.world_scale + SWARM_EDGE_MARGIN,
                    self.rng.uniform(SWARM_EDGE_MARGIN, self.world_scale - SWARM_EDGE_MARGIN)
                ]


            if self.is_far_from_other_swarms(center):
                break
            else:
                continue

        positions = []

        while len(positions) < self.num_drones_per_swarm:
            x = self.rng.uniform(center[0]- SWARM_RADIUS, center[0] + SWARM_RADIUS)
            y = self.rng.uniform(center[1] - SWARM_RADIUS, center[1] + SWARM_RADIUS)
            z = self.rng.uniform(center[2] - SWARM_RADIUS, center[2] + SWARM_RADIUS)

            candidate = np.array([x,y,z])

            # enforce min separation
            valid = True
            for p in positions:
                if np.linalg.norm(candidate - p) < SWARM_MIN_SEPARATION:
                    valid = False
                    break

            if valid:
                positions.append(candidate)

        return center, np.array(positions, dtype=np.float32)


    def make_valid_target(self, color, min_dist = 0.5, max_attempts = 1000) -> Target:
        """
        Make a valid target for all drones, i.e. a target that does not collide with the drone itself, other obstacles
        or other drones.
        """
        for _ in range(max_attempts):
            t = self.generate_xyz()

            drone_positions = self._spawn_positions
            possible_target = Target(color,t, GOAL_RADIUS, physics_call= self._call_physics, call_gui=self._call_gui)
            if possible_target.isValidPosition(drone_positions, self._obstacles, min_dist):
                possible_target.spawn_target()
                return possible_target
            else:
                continue

        backup = self.generate_xyz()

        return Target(backup, GOAL_RADIUS, physics_call = self._call_physics, call_gui = self._call_gui)


    def _make_targets(self) -> list[Target]:
        """ Make a target for all drones."""
        targets = []


        if self.num_swarms == 1:
            t = self.make_valid_target(COLORS[0])
            # repeat shared target for all drones
            targets = [t] * self.total_num_drones
            return targets
        else:
            for i in range(self.num_swarms):
                t = self.make_valid_target(COLORS[i])

                #TODO perhaps change this logic, self._targets stores it in a dumb way
                # List that contain the target per drone
                drone_targets = [t] * self.num_drones_per_swarm
                targets.extend(drone_targets)

            return targets

        #TODO add logic for when more than 1 target exists
        return targets


    def _spawn_obstacles(self) -> None:
        """ Spawn the obstacles. """

        rng = np.random.default_rng()

        self._obstacles = []

        if self.num_static_obstacles + self.num_dynamic_obstacles <= 0 or self._p is None:
            return
        rng = self.rng

        # A max number of attempts for creating an obstacle
        max_attempts = 10000
        for _ in range(self.num_static_obstacles):
            shape = rng.choice(SUPPORTED_SHAPES)

            for _ in range(max_attempts):

                candidate = self.generate_xyz()

                # Check that it is far away from drones
                if not self.is_far_from_drones(candidate, MIN_SPAWN_DIST_FROM_DRONE):
                    continue

                ori = [float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(-np.pi, np.pi))]
                size = float(rng.uniform(0.8, 1.2))

                obj = EnvObject(candidate, size, ori)
                body = obj.create(shape, physics_call=self._call_physics)

                # Check if the obstacle is overlapping with any other obstacle
                if (self.obstacle_overlap(obj)):
                    self._call_physics(self._p.removeBody, body)
                    continue

                self._obstacles.append(obj)
                break

        for _ in range(self.num_dynamic_obstacles):
            axis = rng.choice([0, 1, 2])
            shape = rng.choice(SUPPORTED_SHAPES)

            for _ in range(max_attempts):

                candidate = self.generate_xyz()

                # Check if obstacle is far enough from drones, to prevent spawn killing
                if not self.is_far_from_drones(candidate, MIN_SPAWN_DIST_FROM_DRONE):
                    continue

                displacement = rng.uniform(1, 3)
                min_pos = candidate[axis] - displacement
                max_pos = candidate[axis] + displacement

                # Check if obstacle is going beyond world borders, if so generate another obstacle
                if self.out_of_bounds(max_pos, min_pos, axis):
                    continue

                # Generate other attributes with some variability
                ori = [float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(-np.pi, np.pi))]
                size = float(rng.uniform(0.8, 1.2))
                speed = self.rng.uniform(0.8, 1.2)
                direction = self.rng.choice([-1,1])

                obj = Dynamic(axis,min_pos,max_pos,speed,direction,candidate, size, ori)
                body = obj.create(shape, physics_call= self._call_physics)

                # Check if the obstacle is overlapping with any other obstacle
                if (self.obstacle_overlap(obj)):
                    self._call_physics(self._p.removeBody, body)
                    continue
                self._obstacles.append(obj)
                break


    def _update_obstacles(self) -> None:
        """ Update the dynamic obstacles. """
        if (
            self._p is None
            or len(self._obstacles) == 0
        ):
            return

        t = 1.0 / 60.0

        for obstacle in self._obstacles:
            if isinstance(obstacle, Dynamic):
                obstacle.update_obstacle(t)


    def obstacle_overlap(self, object):
        """ Helper function to see if objects being created overlap with another object. """
        for obs in self._obstacles:
            if object.isOverlapping(obs):
                return True

        return False


    def out_of_bounds(self, max_pos, min_pos, axis):
        """ Helper function used to see if dynamic objects are going out of bounds. """
        if axis != 2:
            if max_pos >= self.world_scale or min_pos <= -self.world_scale:
                return True
            else:
               return False
        else:
            if max_pos >= self.world_scale or min_pos <= 0:
                return True
            else:
                return False

    def drone_out_of_bounds(self, pos:np.ndarray) -> bool:
        """ Helper function to see if drones are going out of bounds (considered as a collision)."""
        x,y,z = float(pos[0]), float(pos[1]), float(pos[2])

        return (
            x <= -self.world_scale
            or x >= self.world_scale
            or y <= -self.world_scale
            or y >= self.world_scale
            or z <= 0.0
            or z >= self.world_scale
        )


    def generate_xyz(self):
        """ Generate the starting position. """
        x = self.rng.uniform(-self.world_scale, self.world_scale)
        y = self.rng.uniform(-self.world_scale, self.world_scale)
        z = self.rng.uniform(0, self.world_scale)

        return np.array([x,y,z], dtype=np.float32)


    def is_far_from_other_swarms(self, candidate_center):
        if not self.drones_in_swarm:
            return True
        else:
            for swarm_info in self.drones_in_swarm.values():

                d = np.linalg.norm(np.array(candidate_center)- np.array(swarm_info["center"]))
                if d <= SWARM_RADIUS:
                    return False
        return True

    def is_far_from_drones(self, point: np.ndarray, min_dist:float) -> bool:
        """ Helper function to determine if potential obstacle spawn positions is far from the drone spawn positions."""
        if self._spawn_positions is None:
            return True
        d = np.linalg.norm(self._spawn_positions - point, axis=1)

        return np.all(d >= min_dist)


    def despawn_drone_in_gui(self, drone_idx: int) -> None:
        """Helper function to despawn a drone if it reached the target. """
        if self._p is None or not (0 <= drone_idx < len(self._drone_ids)):
            return

        p = self._p
        body_id = self._drone_ids[drone_idx]

        # "despawning" happens by moving drone far away
        far_position = [1000.0 + 10.0 * drone_idx, 1000.0, -100.0]
        far_orientation = [0.0, 0.0, 0.0, 1.0]

        try:
            self._call_physics(
                p.resetBasePositionAndOrientation,
                body_id,
                far_position,
                far_orientation,
            )
        except:
            pass

        try:
            self._call_physics(
                p.resetBaseVelocity,
                body_id,
                [0.0, 0.0, 0.0], # make drone stop moving at far position
                [0.0, 0.0, 0.0],
            )
        except Exception:
            pass

        # disable collisions against everything
        try:
            self._call_physics(
                p.setCollisionFilterGroupMask,
                body_id, -1, 0, 0)
        except Exception:
            pass

        # make drone invisible on gui
        try:
            num_joints = self._call_physics(
                p.getNumJoints,
                body_id)
        except Exception:
            num_joints = 0

        for link_idx in range(-1, num_joints):
            try:
                self._call_physics(
                    p.changeVisualShape,
                    body_id,
                    link_idx,
                    rgbaColor=[1.0, 1.0, 1.0, 0.0],
                )
            except Exception:
                pass

        # remove lidar lines for despawned drone
        for lid in self._lidar_line_ids.get(drone_idx, []):
            self._call_gui(p.removeUserDebugItem, lid)
        self._lidar_line_ids[drone_idx] = []


    def despawn_newly_reached_drones(self, just_reached: np.ndarray) -> None:
        """ Helper function to despawn a drone if it reached the target. """
        for i in range(len(just_reached)):
            if bool(just_reached[i] and not bool(self._despawned[i])):
                self.despawn_drone_in_gui(i)
                self._despawned[i] = True


    def despawn_newly_collided_drones(self, collisions: np.ndarray) -> None:
        """ Helper function to despawn a drone if it collided with an obstacle and drone. """
        for i in range(len(collisions)):
            if bool(collisions[i] and not bool(self._despawned[i])):
                self.despawn_drone_in_gui(i)
                self._despawned[i] = True


    # ----------------------------- OBSERVATION / ENVIRONMENT RETRIEVAL METHODS -----------------------------

    def _build_obs_dict(self) -> Dict[str, np.ndarray]:
        """Build a dictionary containing every drone's observation space vector"""

        obs_dict: Dict[str, np.ndarray] = {}

        # if drone does not yet have target, create one
        if self._targets is None:
            self._targets = self._make_targets()

        # compute drone positions and swarm centre
        positions = self.active_drone_positions()
        centroid = self._swarm_centroid(positions)

        for i, a in enumerate(self.possible_agents):

            # check if agent already despawned
            if self._despawned[i]:
                obs_dict[a] = np.zeros((self._base_obs_dim + self._lidar_dim,), dtype=np.float32)
                continue

            # if drone not active, give it observation zero and skip the rest
            if i >= self.total_num_drones:
                obs_dict[a] = np.zeros((self._base_obs_dim + self._lidar_dim,), dtype=np.float32)
                continue

            # get raw state vector from simulator, and extract position and velocity
            state = self._env._getDroneStateVector(i)
            pos = np.array(state[POS_SLICE], dtype=np.float32)
            vel = np.array(state[10:13], dtype=np.float32) # TODO IS VEL 10:13?

            # compute distance to assigned target
            target = self._targets[i].start_pos
            to_target = (target - pos).astype(np.float32)

            # compute distance to nearest obstacle, swarm centre and drone
            to_obs = self._nearest_obstacle_vector(pos)
            to_centroid = (centroid - pos).astype(np.float32)
            to_nearest_teammate = self._nearest_active_teammate_vector(i)

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


    def all_drone_positions(self) -> np.ndarray:
        """ Retrieve all the drone positions (whether dead, despawned or active). """
        positions = np.zeros((self.total_num_drones, 3), dtype=np.float32)

        for i in range(self.total_num_drones):
            state = self._env._getDroneStateVector(i)
            positions[i] = np.array(state[0:3], dtype=np.float32)

        return positions


    def active_drone_mask(self) -> np.ndarray:
        """ Drones that should still influence the ongoing episodes (i.e. not dead or despawned)."""
        return ~(self._dead | self._reached_once | self._despawned)


    def active_drone_indices(self) -> list[int]:
        """ Retrieve all active drone indices."""
        return [i for i, active in enumerate(self.active_drone_mask()) if active]


    def active_drone_positions(self) -> np.ndarray:
        """ Retrieve all active drone positions."""
        idx = self.active_drone_indices()

        if len(idx) == 0:
            return np.zeros((0,3), dtype=np.float32)

        positions = np.zeros((len(idx), 3), dtype=np.float32)
        for k, i in enumerate(idx):
            state = self._env._getDroneStateVector(int(i))
            positions[k] = np.array(state[POS_SLICE], dtype=np.float32)

        return positions


    @staticmethod
    def _swarm_centroid(positions: np.ndarray) -> np.ndarray:
        """ Compute the swarm centroid. """
        if positions.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)

        return positions.mean(axis=0).astype(np.float32)


    def _nearest_active_teammate_vector(self, i: int) -> np.ndarray:
        """ Retrieve the vector to the nearest teammate to drone i."""
        if self._dead[i] or self._reached_once[i] or self._despawned[i]:
            return np.zeros(3, dtype=np.float32)

        state_i = self._env._getDroneStateVector(i)
        pos_i = np.array(state_i[POS_SLICE], dtype=np.float32)

        candidate_neighbours = []
        for j in range(self.total_num_drones):
            if j == i:
                continue
            if self._dead[j] or self._reached_once[j] or self._despawned[j]:
                continue
            state_j = self._env._getDroneStateVector(j)
            pos_j = np.array(state_j[POS_SLICE], dtype=np.float32)
            candidate_neighbours.append(pos_j)

        if len(candidate_neighbours) == 0:
            return np.zeros(3, dtype=np.float32)

        candidate_neighbours = np.array(candidate_neighbours, dtype=np.float32)
        d = candidate_neighbours - pos_i[None, :]
        dist = np.linalg.norm(d, axis=1) + 1e-8

        # return vector distance to nearest drone
        return d[int(np.argmin(dist))].astype(np.float32)


    def _nearest_obstacle_vector(self, drone_pos: np.ndarray) -> np.ndarray:
        """ Retrieve the vector to the nearest obstacle. """

        if self._p is None or len(self._obstacles) == 0:
            # if there are no obstacles, set the distance really far away
            return np.array([LIDAR_RANGE * 2.0, 0.0, 0.0], dtype=np.float32) # TODO PEDRO I KEPT THIS ONE SINCE IF ITS ZERO ITS A COLLISION

        p = self._p
        obs_pos = []
        for object in self._obstacles:
            try:
                pos, _ = self._call_physics(p.getBasePositionAndOrientation, object.body_id)
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


    def check_collisions_per_drone(self):
        """
        Returns:
            collided[i] is True if drone i collided with anything:
            - obstacle
            - another drone
            - world boundary (wall)

        Also returns the specific collision types
        """
        wall_collisions = np.zeros(self.total_num_drones, dtype=np.bool_)
        obstacle_collisions = np.zeros(self.total_num_drones, dtype=np.bool_)
        drone_collisions = np.zeros(self.total_num_drones, dtype=np.bool_)

        if self._p is None or len(self._drone_ids) == 0:
            collided = wall_collisions | obstacle_collisions | drone_collisions
            return collided, wall_collisions, obstacle_collisions, drone_collisions

        p = self._p
        active_ids = self._drone_ids[: self.total_num_drones]

        thresh_obstacle = 0.01
        thresh_drone = 0.01

        # drone vs wall
        for i in range(self.total_num_drones):
            if self._dead[i] or self._reached_once[i] or self._despawned[i]:
                continue

            try:
                state = self._env._getDroneStateVector(i)
                pos_i = np.array(state[POS_SLICE], dtype=np.float32)
                if self.drone_out_of_bounds(pos_i):
                    wall_collisions[i] = True
            except Exception:
                continue

        # drone vs obstacle
        if len(self._obstacles) >0:
            for i, did in enumerate(active_ids):
                if self._dead[i] or self._reached_once[i] or self._despawned[i]:
                    continue

                for object in self._obstacles:
                    try:
                        pts = self._call_physics(p.getClosestPoints, bodyA=did, bodyB=object.body_id,
                                                 distance=thresh_obstacle)
                        if pts and len(pts) > 0:
                            obstacle_collisions[i] = True
                            break
                    except Exception:
                        continue

        # drone vs drone
        for i, did1 in enumerate(active_ids):
            if self._dead[i] or self._reached_once[i] or self._despawned[i]:
                continue

            for j, did2 in enumerate(active_ids[i+1:], start = i+1):
                if self._dead[j] or self._reached_once[j] or self._despawned[j]:
                    continue
                try:
                    pts = self._call_physics(p.getClosestPoints, bodyA=did1, bodyB=did2,
                                             distance=thresh_drone)
                    if pts and len(pts) > 0:
                        drone_collisions[i] = True
                        drone_collisions[j] = True
                except Exception:
                    continue

        collided = wall_collisions | obstacle_collisions | drone_collisions
        return collided, wall_collisions, obstacle_collisions, drone_collisions


    @staticmethod
    def _pairwise_sep_stats(positions: np.ndarray) -> Tuple[float, float, bool, bool]:
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



    # ----------------------------- LIDAR METHODS -----------------------------

    @staticmethod
    def _precompute_lidar_dirs_body() -> np.ndarray:

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


    # ----------------------------- STEP FUNCTION -----------------------------
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

        # 2. Freeze drones that already reached the goal:
        freeze_inactive_actions(self, actions)

        # 3. Convert normalised policy actions into simulator (VelocityAviary) velocity commands
        sim_actions = convert_actions_for_simulator(self, actions)

        # 4. Step the simulator using the action
        self._env.step(sim_actions)

        # 5. Build the observation dictionary
        obs_dict = self._build_obs_dict()

        # 6. Global episode state (collision, swarm cohesion, time limit)
        step_state = compute_step_state(self, obs_dict, actions)

        # 7. Compute rewards (per agent)
        reward_dict = compute_rewards(self, obs_dict, sim_actions, step_state)
        term_dict, trunc_dict = compute_terminations(self, step_state)

        # 8. Info dict (for callbacks & logging)
        info_dict = build_info_dict(self, step_state)

        # despawn newly reached drones
        self.despawn_newly_reached_drones(step_state.just_reached)

        self.agents = self.possible_agents[:]
        self.stats.record_step(
            reward_dict,
            self.possible_agents,
            self.total_num_drones,
            step_state,
            self.active_drone_mask()
        )

        return obs_dict, reward_dict, term_dict, trunc_dict, info_dict


    # ----------------------------- RESET ENVIRONMENT -----------------------------

    def write_ep_to_csv(self):
        """ Write episode stats to csv """
        self.stats.write_episode(self._step_count)


    def reset(self, *, seed=None, options=None):
        """ Reset the environment. Save episode stats in csv file for debugging. """

        if self._step_count > 0:
            self.write_ep_to_csv()

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._step_count = 0
        self.agents = self.possible_agents[:]

        # remove episode-specific GUI/custom bodies from previous episode
        self.clear_episode_artifacts()

        # if underlying VelocityAviary client is disconnected, rebuild it
        if not self._pybullet_client_alive():
            self._rebuild_aviary()

        self._env.reset()

        self._init_pybullet_handles()
        self._refresh_drone_ids_from_pybullet()
        self.restore_drone_visuals_and_collisions()

        center, self.spawn_positions = self.generate_drone_swarm_spawn()
        self.set_drone_spawn_positions_in_aviary(self.spawn_positions)

        # reset episode bookkeeping
        self._dead[:] = False
        self._reached_once = np.zeros((self.total_num_drones,), dtype=np.bool_)
        self._despawned = np.zeros((self.total_num_drones,), dtype=np.bool_)

        self.stats.reset_episode()

        self._prev_dists = np.zeros((self.total_num_drones,), dtype=np.float32)
        self._last_reward = np.zeros(self.total_num_drones, dtype=np.float32)
        self._status = np.zeros(self.total_num_drones, dtype=np.int32)

        self._wall_collision_once[:] = False
        self._obstacle_collision_once[:] = False
        self._drone_collision_once[:] = False

        # refresh episode content
        self._spawn_obstacles()
        self._targets = self._make_targets()

        obs0 = self._build_obs_dict()

        for i in range(self.total_num_drones):
            pos = obs0[self.possible_agents[i]][POS_SLICE]
            tgt = self._targets[i]
            self._prev_dists[i] = float(np.linalg.norm(tgt.start_pos - pos))

        info_dict = {a: {} for a in self.agents}

        return obs0, info_dict


    def clear_episode_artifacts(self) -> None:
        """Helper for resetting episode artifacts."""
        self._clear_debug_visuals()
        self.remove_all_obstacles()
        self.remove_targets()


    def set_drone_spawn_positions_in_aviary(self, spawn_positions: np.ndarray) -> None:
        """ Send the drone swarm to their new spawn position per episode."""
        if self._p is None:
            return

        p = self._p
        default_orientation = [0.0, 0.0, 0.0, 1.0]

        for i, body_id in enumerate(self._drone_ids[:self.total_num_drones]):
            pos = spawn_positions[i].tolist()

            try:
                self._call_physics(
                    p.resetBasePositionAndOrientation,
                    body_id,
                    pos,
                    default_orientation,
                )
            except Exception:
                pass

            try:
                self._call_physics(
                    p.resetBaseVelocity,
                    body_id,
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                )
            except Exception:
                pass


    def remove_all_obstacles(self) -> None:
        """Helper for removing obstacles from current episode environment."""
        if self._p is None:
            self._obstacles = []
            return

        for obj in self._obstacles:
            try:
                if getattr(obj, "body_id", None) is not None:
                    self._call_physics(self._p.removeBody, obj.body_id)
            except Exception:
                pass

        self._obstacles = []


    def remove_targets(self) -> None:
        """Helper for removing targets from current episode environment."""
        if self._p is None or not self._targets:
            self._targets = []
            return

        seen = set()
        for tgt in self._targets:
            if tgt is None:
                continue

            body_id = getattr(tgt, "body_id", None)
            if body_id is None or body_id in seen:
                continue

            seen.add(body_id)
            try:
                self._call_physics(self._p.removeBody, body_id)
            except Exception:
                pass

        self._targets = []


    def restore_drone_visuals_and_collisions(self) -> None:
        """ Helper for restoring visuals and collisions from previously despawned drones. """
        if self._p is None:
            return

        p = self._p

        for i, body_id in enumerate(self._drone_ids[:self.total_num_drones]):
            color_for_drone = COLORS[i//self.num_drones_per_swarm]
            try:
                self._call_physics(
                    p.setCollisionFilterGroupMask,
                    body_id, -1, 1, 1)
            except Exception:
                pass

            try:
                num_joints = self._call_physics(
                    p.getNumJoints,
                    body_id)
            except Exception:
                num_joints = 0

            for link_idx in range(-1, num_joints):
                try:
                    self._call_physics(
                        p.changeVisualShape,
                        body_id,
                        link_idx,
                        rgbaColor = color_for_drone,
                    )
                except Exception:
                    pass


    def _clear_debug_visuals(self) -> None:
        if self._p is None or self.render_mode != "human":
            self._lidar_line_ids = {}
            return

        p = self._p

        for _, line_ids in list(self._lidar_line_ids.items()):
            for lid in line_ids:
                self._call_gui(p.removeUserDebugItem, lid)
        self._lidar_line_ids = {}

    def action_space(self, agent):
        """ Retrieve the action space """
        return self._action_space

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

