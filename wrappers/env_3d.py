from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from envs.createObjects import Target, EnvObject, Dynamic
from wrappers.collisionQuery import CollisionQueryWorld
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
            max_drones_per_swarm: Optional[int] = None,
            num_swarms: int = 1,
            lidar: bool = False,
            render_mode: Optional[str] = None,
            seed: Optional[int] = None,
    ):

        self._init_config(
            world_scale=world_scale,
            num_drones_per_swarm=num_drones_per_swarm,
            num_static_obstacles=num_static_obstacles,
            num_dynamic_obstacles=num_dynamic_obstacles,
            max_steps=max_steps,
            max_drones_per_swarm=max_drones_per_swarm,
            num_swarms=num_swarms,
            lidar=lidar,
            render_mode=render_mode,
        )

        self._init_rng(seed)
        self._init_agents()
        self._init_episode_state()
        self._init_simulator()
        self._init_query_world()
        self._init_spaces()
        self._init_logging()


    def _init_config(
            self,
            world_scale: float,
            num_drones_per_swarm: int,
            num_static_obstacles: int,
            num_dynamic_obstacles: int,
            max_steps: int,
            max_drones_per_swarm: Optional[int],
            num_swarms: int,
            lidar: bool,
            render_mode: Optional[str],
    ):
        """ Store environment configuration. """
        self.world_scale = float(world_scale)
        self.num_swarms = int(num_swarms)

        # active drones for the current curriculum stage
        self.num_drones_per_swarm = int(num_drones_per_swarm)

        # fixed capacity across all curriculum stages
        if max_drones_per_swarm is None:
            max_drones_per_swarm = num_drones_per_swarm
        self.max_drones_per_swarm = int(max_drones_per_swarm)

        if self.max_drones_per_swarm < self.num_drones_per_swarm:
            raise ValueError(
                f"max_drones_per_swarm ({self.max_drones_per_swarm}) "
                f"must be >= num_drones_per_swarm ({self.num_drones_per_swarm})"
            )

        # active count this stage
        # ensure at least one drone in environment
        self.total_num_drones = max(
            1,
            self.num_drones_per_swarm * self.num_swarms
        )

        # fixed count exposed to PettingZoo/SB3
        self.max_total_num_drones = max(1, self.max_drones_per_swarm * self.num_swarms)

        # Spawn position of every drone
        self.spawn_positions = np.empty((0,3))

        # Dict with {swarm: [position of drone]}
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
        self.possible_agents = [f"drone_{i}" for i in range(self.max_total_num_drones)] # fixed list of all drone names in env
        self.agents = self.possible_agents[:]  # current active list in episode (used to control dead / despawned drones)


    def _init_episode_state(self) -> None:
        """ Initialise per-episode bookkeeping. """
        # boolean array representing death status of each drone
        self._dead = np.zeros(self.max_total_num_drones, dtype=np.bool_)

        # boolean array of target reached memory: once reached, stays True until reset
        self._reached_once = np.zeros((self.max_total_num_drones,), dtype=np.bool_)
        self._despawned = np.zeros((self.max_total_num_drones,), dtype=np.bool_)

        # target for each drone; self._targets[i] belongs to drone i
        self._targets: Optional[list[Target]] = None

        # distance of each drone to its target in previous step (used for reward computation)
        self._prev_dists = np.zeros(self.max_total_num_drones, dtype=np.float32)
        # reward value of each drone in previous step
        self._last_reward = np.zeros(self.max_total_num_drones, dtype=np.float32)

        self._wall_collision_once = np.zeros((self.max_total_num_drones,), dtype=np.bool_)
        self._obstacle_collision_once = np.zeros((self.max_total_num_drones,), dtype=np.bool_)
        self._drone_collision_once = np.zeros((self.max_total_num_drones,), dtype=np.bool_)

        self._step_count = 0


    def _init_simulator(self) -> None:
        """ Create drone simulator and PyBullet handles. """
        # list of drone's initial position
        self.spawn_positions = np.empty((0, 3), dtype=np.float32)
        self.drones_in_swarm = {}

        for i in range(self.num_swarms):
            edge, center, drones = self.generate_drone_swarm_spawn()
            self.drones_in_swarm[i] = {"edge": edge, "center": center, "drones":drones}
            self.spawn_positions = np.vstack([self.spawn_positions, drones])

        self.spawn_positions = self.spawn_positions.astype(np.float32)

        n_missing = self.max_total_num_drones - self.spawn_positions.shape[0]
        if n_missing > 0:
            dummy_positions = []
            for k in range(n_missing):
                dummy_positions.append([1000.0 + 10.0 * k, 1000.0, -100.0])
            dummy_positions = np.array(dummy_positions, dtype=np.float32)
            # pad rest with dummy positions far away so aviary gets exactly max_total_num_drones initial positions
            self.spawn_positions = np.vstack([self.spawn_positions, dummy_positions])

        self._create_aviary()
        self._targets = self._make_targets()

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

        # Precompute LiDAR ray directions once, regardless of whether lidar is enabled
        # in this stage. This keeps shapes and helper logic consistent.
        self._lidar_dirs_body = self._precompute_lidar_dirs_body()


    def _init_logging(self) -> None:
        """ Create CSV log file if missing. """

        self.stats = EpisodeStatsTracker("episode_stats.csv")


    def _init_pybullet_handles(self) -> None:
        self._gui_client_id = None
        if self.render_mode == "human":
            self._gui_client_id = self._client_id
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
        if self._p is None:
            return None
        pcid = self._gui_client_id  # use stored ID, not max()
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
            self._env.close()
        except Exception:
            pass

        self._create_aviary()


    def _create_aviary(self):
        """Create VelocityAviary from scratch."""
        self._env = VelocityAviary(
            drone_model=DroneModel.CF2X,
            num_drones=self.max_total_num_drones,
            initial_xyzs=self.spawn_positions,
            physics=Physics.PYB,
            gui=(self.render_mode == "human"),
            record=False,
            user_debug_gui=False,
        )
        self._env.SPEED_LIMIT = 3.0

        self._p = None
        self._client_id = None
        self._drone_ids = []
        self._obstacles = []
        self._lidar_line_ids = {}
        self._wall_ids = []

        self._init_pybullet_handles()
        self._refresh_drone_ids_from_pybullet()


    # ----------------------------- (DE)SPAWN DRONES, OBSTACLES AND TARGETS -----------------------------


    def generate_drone_swarm_spawn(self):
        max_attempts = 1000
        edge = 0

        for i in range(max_attempts):
            edge = self.rng.integers(0, 4)
            center = self.generate_swarm_center_xyz(edge, SWARM_EDGE_MARGIN)


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

        return edge, center, np.array(positions, dtype=np.float32)


    def generate_edge_xyz(self, edge, margin):
        """Generate a random position on a given world edge."""

        if edge == 0:  # +X edge
            pos = [
                self.world_scale - margin,
                self.rng.uniform(-self.world_scale + margin, self.world_scale - margin),
                self.rng.uniform(margin, self.world_scale - margin)
            ]

        elif edge == 1:  # -X edge
            pos = [
                -self.world_scale + margin,
                self.rng.uniform(-self.world_scale + margin, self.world_scale - margin),
                self.rng.uniform(margin, self.world_scale - margin)
            ]

        elif edge == 2:  # +Y edge
            pos = [
                self.rng.uniform(-self.world_scale + margin, self.world_scale - margin),
                self.world_scale - margin,
                self.rng.uniform(margin, self.world_scale - margin)
            ]

        else:  # -Y edge
            pos = [
                self.rng.uniform(-self.world_scale + margin, self.world_scale - margin),
                -self.world_scale + margin,
                self.rng.uniform(margin, self.world_scale - margin)
            ]

        return np.array(pos, dtype=np.float32)


    # Generates the swarm center
    def generate_swarm_center_xyz(self, edge, margin):
        return self.generate_edge_xyz(edge, margin)


    # Generate the spawn position of target, opposite to the swarms center
    def generate_target_xyz(self, edge, margin):
        opposite_edge = {0: 1, 1: 0, 2: 3, 3: 2}[edge]
        return self.generate_edge_xyz(opposite_edge, margin)


    def _spawn_walls(self) -> None:

        if self._p is None:
            return

        s = self.world_scale

        positions = [
            ("z_pos_wall", [0, 0, s]),  # ceiling
            ("z_pos_wall", [0, 0, -s]),  # floor
            ("x_pos_wall", [s, 0, 0]),  # front
            ("x_pos_wall", [-s, 0, 0]),  # back
            ("y_pos_wall", [0, s, 0]),  # right
            ("y_pos_wall", [0, -s, 0]),  # left
        ]

        objects = [(key, EnvObject(pos, s, [0, 0, 0])) for key, pos in positions]
        self._wall_ids = [obj.create(key, self._call_physics, color=(0.5, 0.5, 0.5, 1.0)) for key, obj in objects]


    def make_valid_target(self,swarm_edge, color, min_dist = 0.5, max_attempts = 1000) -> Target:
        """
        Make a valid target for all drones, i.e. a target that does not collide with the drone itself, other obstacles
        or other drones.
        """
        for _ in range(max_attempts):
            t = self.generate_target_xyz(swarm_edge, GOAL_EDGE_MARGIN)

            drone_positions = self.spawn_positions[:self.total_num_drones]
            possible_target = Target(color,t, GOAL_RADIUS, physics_call= self._call_physics, call_gui=self._call_gui)
            if possible_target.isValidPosition(drone_positions, self._obstacles, min_dist):
                possible_target.spawn_target()
                return possible_target
            else:
                continue

        backup = self.generate_xyz()

        return Target(color, backup, GOAL_RADIUS,
                      physics_call=self._call_physics,
                      call_gui=self._call_gui)


    def _make_targets(self) -> list[Target]:
        """ Make a target for all drones."""
        targets = []

        if self.num_swarms == 1:
            swarm_edge = self.drones_in_swarm[0]["edge"]
            t = self.make_valid_target(swarm_edge,COLORS[0])
            # repeat shared target for all drones
            targets = [t] * self.total_num_drones
            return targets
        else:
            for i in range(self.num_swarms):
                swarm_centroid = self.drones_in_swarm[i]["center"]
                swarm_edge = self.drones_in_swarm[i]["edge"]

                t = self.make_valid_target(swarm_edge,COLORS[i])

                # List that contain the target per drone
                drone_targets = [t] * self.num_drones_per_swarm
                targets.extend(drone_targets)

            return targets


    def _spawn_obstacles(self):
        """ Spawn the obstacles. """

        self._obstacles = []


        if self.num_static_obstacles + self.num_dynamic_obstacles <= 0 or self._p is None:
            return

        # A max number of attempts for creating an obstacle
        swarm_list = list(self.drones_in_swarm.keys())
        max_attempts = 100000

        for _ in range(self.num_dynamic_obstacles):
            axis = self.rng.choice([0, 1, 2])
            shape = self.rng.choice(SUPPORTED_SHAPES)
            self._spawn_dynamic_obstacles(max_attempts, shape, axis)
            # self._rebuild_query_world_from_current_scene()

        for _ in range(self.num_static_obstacles):
            shape = self.rng.choice(SUPPORTED_SHAPES)
            swarm_list = self._spawn_static_obstacles(max_attempts, shape, swarm_list)


    def _spawn_static_obstacles(self, max_attempts, shape, swarm_list) -> None:
        """ Spawn the obstacles. """
        chosen_swarm = None
        blocking_obstacles_chance = self.rng.uniform(0.0, 1.0)

        for _ in range(max_attempts):
            if blocking_obstacles_chance < 0.5 and len(swarm_list) >= 1:
                chosen_swarm = self.rng.choice(swarm_list)
                obj = self._spawn_blocking_obstacles(chosen_swarm)
            else:
                candidate = self.generate_xyz()
                ori = [float(self.rng.uniform(-np.pi, np.pi)), float(self.rng.uniform(-np.pi, np.pi)), float(self.rng.uniform(-np.pi, np.pi))]
                size = float(self.rng.uniform(0.8, 1.2))

                obj = EnvObject(candidate, size, ori)

            if(self.in_swept_area(obj)) :
                continue
            body = obj.create(shape, physics_call=self._call_physics)

            # Check if the obstacle is overlapping with any other obstacle or too close to drones or target
            if self.obstacle_overlap(obj) and obj.isValidPosition(self.spawn_positions[:self.total_num_drones], self._targets, MIN_SPAWN_DIST_FROM_DRONE):
                self._call_physics(self._p.removeBody, body)
                continue

            if chosen_swarm is not None:
                swarm_list.remove(chosen_swarm)
            self._obstacles.append(obj)
            break
        return swarm_list


    def in_swept_area(self, object) :
        for obs in self._obstacles :
            if isinstance(obs, Dynamic) and object.is_in_swept_area(obs) :
                return True
        return False


    def _spawn_dynamic_obstacles(self, max_attempts, shape, axis) -> None:
        """ Spawn the dynamic obstacles. """
        for _ in range(max_attempts):

            candidate = self.generate_xyz()

            displacement = self.rng.uniform(1, 3)
            min_pos = candidate[axis] - displacement
            max_pos = candidate[axis] + displacement

            # Check if obstacle is going beyond world borders, if so generate another obstacle
            if self.out_of_bounds(max_pos, min_pos, axis):
                continue

            # Generate other attributes with some variability
            ori = [float(self.rng.uniform(-np.pi, np.pi)), float(self.rng.uniform(-np.pi, np.pi)), float(self.rng.uniform(-np.pi, np.pi))]
            size = float(self.rng.uniform(0.8, 1.2))
            speed = self.rng.uniform(0.8, 1.2)
            direction = self.rng.choice([-1,1])

            
            obj = Dynamic(axis,min_pos,max_pos,speed,direction,candidate, size, ori)
            obj.set_shape_spec(shape)
            #BRING THIS BACK LATER TODO
            # if self.candidate_collides_over_time(obj,horizon_steps=self.max_steps, dt=1.0/60.0):
            #     continue
            
            body = obj.create(shape, physics_call= self._call_physics)

            # Check if the obstacle is overlapping with any other obstacle and is far enough from targets and drones
            if (self.obstacle_overlap(obj) and obj.isValidPosition(self.spawn_positions[:self.total_num_drones], self._targets, MIN_SPAWN_DIST_FROM_DRONE)):
                self._call_physics(self._p.removeBody, body)
                continue
            self._obstacles.append(obj)
            break


    def _spawn_blocking_obstacles(self, swarm_num) -> EnvObject:
        # A max number of attempts for creating an obstacle

        swarm_center = self.drones_in_swarm[swarm_num]["center"]
        target_pos = self._targets[swarm_num * self.num_drones_per_swarm].start_pos


        candidate = self.generate_blocking_xyz(swarm_center, target_pos)
        ori = [float(self.rng.uniform(-np.pi, np.pi)), float(self.rng.uniform(-np.pi, np.pi)),
               float(self.rng.uniform(-np.pi, np.pi))]
        size = float(self.rng.uniform(1.2, 1.5))

        obj = EnvObject(candidate, size, ori)
        return obj


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
        """ Helper function used to see if entities are going out of bounds. """
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

    
    def _init_query_world(self):
        self._qworld = CollisionQueryWorld()


    def _rebuild_query_world_from_current_scene(self):
        self._qworld.clear()

        for obs in self._obstacles:
            if isinstance(obs, Dynamic):
                self._qworld.add_dynamic_object(id(obs), obs)
            else:
                self._qworld.add_static_object(obs) # TODO CARIM THIS METHOD DOES NOT EXIST IN YOUR CLASS?


    def candidate_collides_over_time(self, candidate, horizon_steps, dt=1.0/60.0, margin=0.0):
        temp_body = self._qworld.add_temp_object(candidate)

        try:
            for step in range(horizon_steps + 1):
                t = step * dt
                cand_pos, cand_quat = candidate.pose_at_time(t)
                self._qworld.set_body_pose(temp_body, cand_pos, cand_quat)

                for obs in self._obstacles:
                    if isinstance(obs, Dynamic):
                        body_id = self._qworld.dynamic_bodies[id(obs)]
                        pos, quat = obs.pose_at_time(t)
                        self._qworld.set_body_pose(body_id, pos, quat)

                for obs in self._obstacles:
                    if isinstance(obs, Dynamic):
                        body_id = self._qworld.dynamic_bodies[id(obs)]
                        if self._qworld.bodies_collide(temp_body, body_id, margin=margin):
                            return True
            return False
        finally:
            self._qworld.remove_body(temp_body)


    def drone_out_of_bounds(self, pos:np.ndarray) -> bool:
        """ Helper function to see if drones are going out of bounds (considered as a collision)."""
        x,y,z = float(pos[0]), float(pos[1]), float(pos[2])

        return (
            x <= -self.world_scale
            or x >= self.world_scale
            or y <= -self.world_scale
            or y >= self.world_scale
            or z <= 0.1 # TODO 0 OR 0.1?
            or z >= self.world_scale
        )


    def generate_xyz(self):
        """ Generate the starting position. """
        x = self.rng.uniform(-self.world_scale, self.world_scale)
        y = self.rng.uniform(-self.world_scale, self.world_scale)
        z = self.rng.uniform(0, self.world_scale)

        return np.array([x,y,z], dtype=np.float32)


    def generate_blocking_xyz(self, swarm_center: np.ndarray, target_pos: np.ndarray,
                              margin: float = 0.1) -> np.ndarray:
        """ Creates an obstacles that blocks the straight line from a center to the target"""
        # Vector from swarm to target
        line_vec = target_pos - swarm_center

        # Pick a random fraction along the line (0=start, 1=end)
        t = self.rng.uniform(0.2, 0.8)  # avoid spawning exactly on swarm or target
        point_on_line = swarm_center + t * line_vec

        # Optional: add a small random perpendicular offset so obstacle is not exactly on line
        # Generate a random vector
        random_offset = self.rng.uniform(-margin, margin, size=3)

        # Project offset perpendicular to the line
        line_dir = line_vec / (np.linalg.norm(line_vec) + 1e-8)  # avoid division by zero
        perp_offset = random_offset - np.dot(random_offset, line_dir) * line_dir

        # Final candidate position
        candidate = point_on_line + perp_offset

        # Clip to world bounds if needed
        candidate = np.clip(candidate, [-self.world_scale, -self.world_scale, 0],
                            [self.world_scale, self.world_scale, self.world_scale])

        return candidate.astype(np.float32)


    def is_far_from_other_swarms(self, candidate_center):
        if not self.drones_in_swarm:
            return True
        else:
            for swarm_info in self.drones_in_swarm.values():

                d = np.linalg.norm(np.array(candidate_center)- np.array(swarm_info["center"]))
                if d <= SWARM_RADIUS:
                    return False
        return True


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
        # if self._targets is None:
        #     print("drone did not have a target")
        #     exit()
        #     self._targets = self._make_targets()

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

            #if self._p is not None and i < len(self._drone_ids) and self._step_count % 100 == 0:
            #    try:
            #        lin_vel_pb, ang_vel_pb = self._call_physics(
            #            self._p.getBaseVelocity,
            #            self._drone_ids[i]
            #        )

            #        lin_vel_pb = np.array(lin_vel_pb, dtype=np.float32)
            #        ang_vel_pb = np.array(ang_vel_pb, dtype=np.float32)

            #        print(f"\n[VEL SEARCH] drone {i} step {self._step_count}")
            #        for start in range(len(state) - 2):
            #            chunk = np.array(state[start:start + 3], dtype=np.float32)
            #            lin_err = np.linalg.norm(chunk - lin_vel_pb)
            #            ang_err = np.linalg.norm(chunk - ang_vel_pb)
            #            print(
            #                f"slice[{start}:{start + 3}] = {chunk} | "
            #                f"lin_err={lin_err:.6f} | ang_err={ang_err:.6f}"
            #            )
            #    except Exception as e:
            #        print("[VEL SEARCH ERROR]", e)

            vel = np.array(state[10:13], dtype=np.float32)

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
            else:
                # Neutral LiDAR values: 1.0 means "no hit within range" for all rays
                scan = np.ones((self._lidar_dim,), dtype=np.float32)

            obs_dict[a] = np.concatenate([base, scan], axis=0).astype(np.float32)

        return obs_dict


    def active_drone_mask(self) -> np.ndarray:
        """ Drones that should still influence the ongoing episodes (i.e. not dead or despawned)."""
        mask = np.zeros(self.max_total_num_drones, dtype=np.bool_)
        mask[:self.total_num_drones] = True
        mask &= ~(self._dead | self._reached_once | self._despawned)
        return mask


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
            return np.array([LIDAR_RANGE * 2.0, 0.0, 0.0], dtype=np.float32)

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
        thresh_wall = 0.02

        # drone vs wall
        for i, did in enumerate(active_ids):
            if self._dead[i] or self._reached_once[i] or self._despawned[i]:
                continue

            # 1) direct contact with wall bodies
            for wid in self._wall_ids:
                try:
                    pts = self._call_physics(p.getContactPoints, bodyA=did, bodyB=wid)
                    if pts and len(pts) > 0:
                        wall_collisions[i] = True
                        break
                except Exception:
                    continue

                try:
                    pts = self._call_physics(
                        p.getClosestPoints,
                        bodyA=did,
                        bodyB=wid,
                        distance=thresh_wall,
                    )
                    if pts and len(pts) > 0:
                        wall_collisions[i] = True
                        break
                except Exception:
                    continue

            # 2) fallback numeric boundary check
            if not wall_collisions[i]:
                try:
                    state = self._env._getDroneStateVector(i)
                    pos_i = np.array(state[POS_SLICE], dtype=np.float32)
                    if self.drone_out_of_bounds(pos_i):
                        wall_collisions[i] = True
                except Exception:
                    pass

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


    def _lidar_scan(self, drone_idx: int, pos_unused: np.ndarray) -> np.ndarray:

        if (not self.lidar) or (self._p is None) or (self._lidar_dim <= 0):
            return np.ones((self._lidar_dim,), dtype=np.float32)

        p = self._p

        did = self._drone_ids[drone_idx] if (0 <= drone_idx < len(self._drone_ids)) else None
        if did is None:
            return np.ones((self._lidar_dim,), dtype=np.float32)

        try:
            base_pos, orn = self._call_physics(p.getBasePositionAndOrientation, did)
            origin = np.array(base_pos, dtype=np.float32)
            R = self._rotmat_from_quat(np.array(orn, dtype=np.float32))
        except Exception:
            return np.ones((self._lidar_dim,), dtype=np.float32)

        dirs_world = (R @ self._lidar_dirs_body.T).T
        dirs_world = dirs_world.astype(np.float32)

        # move ray start slightly outside the drone body along each ray direction
        LIDAR_START_OFFSET = 0.15
        ray_from = origin[None, :] + dirs_world * LIDAR_START_OFFSET
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
                hit_body = int(r[0])
                hf = float(r[2])
                hp = r[3]

                # ignore self-hit if it still happens
                if hit_body == did:
                    hit_frac[i] = 1.0
                    hit_pos[i] = ray_to[i]
                else:
                    hit_frac[i] = np.clip(hf, 0.0, 1.0)
                    hit_pos[i] = np.array(hp, dtype=np.float32)
            except Exception:
                pass

        if LIDAR_NOISE_STD > 0.0:
            hit_frac = np.clip(
                hit_frac + self.rng.normal(0.0, LIDAR_NOISE_STD, size=hit_frac.shape).astype(np.float32),
                0.0,
                1.0,
            )

        #if self.render_mode == "human" and LIDAR_SHOW_RAYS_TEST and self._step_count % 50 == 0:

        show_rays = False
        if self.render_mode == "human":
            show_rays = LIDAR_SHOW_RAYS_TRAIN
            # or use TEST flag for non-training scripts if you want separate behavior

        if show_rays and self._step_count % 50 == 0:
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

        # force inactive drones to do nothing
        for i, agent in enumerate(self.possible_agents):
            if i >= self.total_num_drones:
                actions[agent] = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)

        # 3. Convert normalised policy actions into simulator (VelocityAviary) velocity commands
        sim_actions = convert_actions_for_simulator(self, actions)

        self._last_sim_actions = sim_actions.copy()
        # 4. Step the simulator using the action
        self._env.step(sim_actions)

        # 5. Build the observation dictionary
        obs_dict = self._build_obs_dict()

        # 6. Global episode state (collision, swarm cohesion, time limit)
        step_state = compute_step_state(self, obs_dict)

        # 7. Compute rewards (per agent)
        reward_dict = compute_rewards(self, obs_dict, sim_actions, step_state)
        term_dict, trunc_dict = compute_terminations(self, step_state)

        # 8. Info dict (for callbacks & logging)
        info_dict = build_info_dict(self, step_state)

        # despawn newly reached drones
        self.despawn_newly_reached_drones(step_state.reached_now)

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

        self._set_rendering_enabled(False)

        try:
            # remove episode-specific GUI/custom bodies from previous episode
            self.clear_episode_artifacts()

            # fresh episode spawn positions
            self.spawn_positions = np.empty((0, 3), dtype=np.float32)
            self.drones_in_swarm = {}

            for i in range(self.num_swarms):
                edge, center, drones = self.generate_drone_swarm_spawn()
                self.drones_in_swarm[i] = {"edge": edge, "center": center, "drones": drones}
                self.spawn_positions = np.vstack([self.spawn_positions, drones])

            self.spawn_positions = self.spawn_positions.astype(np.float32)

            n_missing = self.max_total_num_drones - self.spawn_positions.shape[0]
            if n_missing > 0:
                dummy_positions = []
                for k in range(n_missing):
                    dummy_positions.append([1000.0 + 10.0 * k, 1000.0, -100.0])
                dummy_positions = np.array(dummy_positions, dtype=np.float32)
                self.spawn_positions = np.vstack([self.spawn_positions, dummy_positions])

            print("drones_in_swarm", self.drones_in_swarm)
            print("spawn_positions", self.spawn_positions)

            # if underlying VelocityAviary client is disconnected, rebuild it
            if not self._pybullet_client_alive():
                self._rebuild_aviary()

            self._env.reset()

            self._init_pybullet_handles()
            self._refresh_drone_ids_from_pybullet()

            self.restore_drone_visuals_and_collisions()

            # Move drones to their new spawn position
            for i, pos in enumerate(self.spawn_positions):
                self._p.resetBasePositionAndOrientation(
                    self._drone_ids[i],
                    pos,  # new spawn position
                    [0, 0, 0, 1]  # neutral orientation (quaternion)
                )
                self._p.resetBaseVelocity(
                    self._drone_ids[i],
                    linearVelocity=[0, 0, 0],
                    angularVelocity=[0, 0, 0]
                )



            # reset episode bookkeeping
            self._dead[:] = False
            self._reached_once[:] = False
            self._despawned[:] = False

            self.stats.reset_episode()

            self._prev_dists[:] = 0.0
            self._last_reward[:] = 0.0
            self._status = np.zeros(self.max_total_num_drones, dtype=np.int32)

            self._wall_collision_once[:] = False
            self._obstacle_collision_once[:] = False
            self._drone_collision_once[:] = False

            # refresh episode content
            self._rebuild_query_world_from_current_scene()
            self._spawn_obstacles()
            self._spawn_walls()
            self._targets = self._make_targets()


            obs0 = self._build_obs_dict()

            for i in range(self.total_num_drones):
                pos = obs0[self.possible_agents[i]][POS_SLICE]
                tgt = self._targets[i]
                self._prev_dists[i] = float(np.linalg.norm(tgt.start_pos - pos))

            info_dict = {a: {} for a in self.agents}
        finally:
            self._set_rendering_enabled(True)

        return obs0, info_dict


    def _set_rendering_enabled(self, enabled: bool) -> None:
        """ Helper to temporarily enable/disable rendering """
        if self._p is None or self.render_mode != "human":
            return
        try:
            self._call_gui(
                self._p.configureDebugVisualizer,
                self._p.COV_ENABLE_RENDERING,
                1 if enabled else 0,
            )
        except Exception:
            pass


    def clear_episode_artifacts(self) -> None:
        """Helper for resetting episode artifacts."""
        self._clear_debug_visuals()
        self.remove_all_obstacles()
        self.remove_targets()
        self.remove_walls()
        # reset spawn positions
        self.spawn_positions = np.empty((0, 3))


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


    def remove_walls(self) -> None:
        """ Helper for removing walls from current episode environment."""
        for body_id in self._wall_ids:
            try:
                self._call_physics(self._p.removeBody, body_id)
            except Exception:
                pass
        self._wall_ids = []


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

        for i, body_id in enumerate(self._drone_ids[:self.max_total_num_drones]):
            swarm_idx = min(i // max(1, self.max_drones_per_swarm), len(COLORS) - 1)
            color_for_drone = COLORS[swarm_idx]
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

    def observation_space(self, agent):
        """Retrieve the observation space."""
        return self._obs_space

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

