# wrappers/pz_parallel_3d.py
from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class DroneSwarm3D(ParallelEnv):
    metadata = {"name": "drone_swarms_3d_v0"}

    def __init__(
        self,
        num_drones: int = 2,
        active_drones: Optional[int] = None,
        num_obstacles: int = 4,
        lidar: bool = False,
        max_steps: int = 500,
        render_mode: Optional[str] = None,
        world_scale: float = 2.0,               #TOO LARGES DRONES DONT REACH TARGET
        seed: Optional[int] = None,
        gravity: float = -9.81,
        # --- motion helpers ---
        vel_scale: float = 15.0,                     #TOO SMALL DRONES DONT REACH TARGET
        takeoff_steps: int = 60,
        takeoff_vz: float = 1.0,
        hover_bias_vz: float = 0.0,
        # --- obstacle params ---
        obstacle_radius: float = 0.30,
        obstacle_height_range: Tuple[float, float] = (0.5, 1.8),
        obstacle_dynamic: bool = False,
        obstacle_drift_amp: float = 0.8,
        obstacle_drift_omega: float = 0.35,
        # --- safety / reward ---
        collision_distance: float = 0.12,
        collision_penalty: float = 200.0,
        near_obstacle_penalty_k: float = 0.6,
        # --- goal / shaping ---
        goal_radius: float = 0.8,
        goal_bonus: float = 80.0,
        progress_k: float = 8.0,
        time_penalty: float = 0.05,
        ctrl_penalty_k: float = 0.003,
        # --- swarm cohesion / separation ---
        swarm_max_sep: float = 2.0,
        swarm_min_sep: float = 0.25,
        cohesion_penalty_k: float = 50.0,
        separation_penalty_k: float = 30.0,
        terminate_on_cohesion_break: bool = False,
        terminate_on_separation: bool = False,
        # --- shared target ---
        shared_target: bool = True,
        # --- lidar params ---
        lidar_num_rays: int = 72,
        lidar_num_layers: int = 5,
        lidar_range: float = 6.0,
        lidar_vertical_fov_deg: float = 45.0,
        lidar_horizontal_fov_deg: float = 360.0,
        lidar_noise_std: float = 0.0,
        lidar_show_rays: bool = True,
        # --- debug ---
        debug: bool = False,
    ):
        self.num_drones = int(num_drones)
        self.active_drones = int(active_drones) if active_drones is not None else self.num_drones
        self.active_drones = max(1, min(self.active_drones, self.num_drones))

        self.num_obstacles = int(num_obstacles)
        self.lidar = bool(lidar)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.world_scale = float(world_scale)
        self.gravity = float(gravity)

        self.vel_scale = float(vel_scale)
        self.takeoff_steps = int(takeoff_steps)
        self.takeoff_vz = float(takeoff_vz)
        self.hover_bias_vz = float(hover_bias_vz)

        self.obstacle_radius = float(obstacle_radius)
        self.obstacle_height_range = (float(obstacle_height_range[0]), float(obstacle_height_range[1]))
        self.obstacle_dynamic = bool(obstacle_dynamic)
        self.obstacle_drift_amp = float(obstacle_drift_amp)
        self.obstacle_drift_omega = float(obstacle_drift_omega)

        self.collision_distance = float(collision_distance)
        self.collision_penalty = float(collision_penalty)
        self.near_obstacle_penalty_k = float(near_obstacle_penalty_k)

        self.goal_radius = float(goal_radius)
        self.goal_bonus = float(goal_bonus)
        self.progress_k = float(progress_k)
        self.time_penalty = float(time_penalty)
        self.ctrl_penalty_k = float(ctrl_penalty_k)

        self.swarm_max_sep = float(swarm_max_sep)
        self.swarm_min_sep = float(swarm_min_sep)
        self.cohesion_penalty_k = float(cohesion_penalty_k)
        self.separation_penalty_k = float(separation_penalty_k)
        self.terminate_on_cohesion_break = bool(terminate_on_cohesion_break)
        self.terminate_on_separation = bool(terminate_on_separation)

        self.shared_target = bool(shared_target)

        self.lidar_num_rays = int(lidar_num_rays)
        self.lidar_num_layers = int(max(1, lidar_num_layers))
        self.lidar_range = float(lidar_range)
        self.lidar_vertical_fov_deg = float(lidar_vertical_fov_deg)
        self.lidar_horizontal_fov_deg = float(lidar_horizontal_fov_deg)
        self.lidar_noise_std = float(lidar_noise_std)
        self.lidar_show_rays = bool(lidar_show_rays)

        self.debug = bool(debug)
        self.rng = np.random.default_rng(seed)

        self.possible_agents = [f"drone_{i}" for i in range(self.num_drones)]
        self.agents = self.possible_agents[:]

        self._step_count = 0
        self._targets: Optional[np.ndarray] = None
        self._prev_dists: Optional[np.ndarray] = None

        self._env = VelocityAviary(
            drone_model=DroneModel.CF2X,
            num_drones=self.num_drones,
            physics=Physics.PYB,
            gui=(render_mode == "human"),
            record=False,
        )

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
        # We will interpret u[:3] as direction (normalized) and u[3] as speed control in [0..1]
        self._action_space = spaces.Box(
            low=np.full((4,), -1.0, dtype=np.float32),
            high=np.full((4,), 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        # base obs:
        # pos(3), vel(3), to_target(3), to_nearest_obstacle(3), to_centroid(3), to_nearest_teammate(3) = 18
        self._base_obs_dim = 18
        self._lidar_dim = (self.lidar_num_rays * self.lidar_num_layers) if self.lidar else 0
        obs_dim = self._base_obs_dim + self._lidar_dim

        self._obs_space = spaces.Box(
            low=np.full((obs_dim,), -1e9, dtype=np.float32),
            high=np.full((obs_dim,), 1e9, dtype=np.float32),
            dtype=np.float32,
        )

        if self.lidar:
            self._lidar_dirs_body = self._precompute_lidar_dirs_body()

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._action_space

    # ---------- pybullet client handling ----------
    def _connected_ids(self) -> List[int]:
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
        ids = self._connected_ids()
        return max(ids) if ids else None

    def _call_physics(self, fn, *args, **kwargs):
        if self._p is None:
            raise RuntimeError("PyBullet not available")
        pcid = self._pcid_for_physics()
        if pcid is not None and "physicsClientId" not in kwargs:
            try:
                return fn(*args, **kwargs, physicsClientId=pcid)
            except Exception:
                pass
        return fn(*args, **kwargs)

    def _call_draw(self, fn, *args, **kwargs):
        if self._p is None:
            return None
        pcid = self._pcid_for_gui_draw()
        try:
            if pcid is not None and "physicsClientId" not in kwargs:
                return fn(*args, **kwargs, physicsClientId=pcid)
            return fn(*args, **kwargs)
        except Exception:
            # silently ignore (prevents the shared-memory "User debug draw failed" spam)
            return None

    # ---------------- helpers ----------------
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
                self._call_physics(p.setGravity, 0, 0, self.gravity)
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

    def _refresh_drone_ids_from_pybullet(self) -> None:
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

    def _make_shared_target(self) -> np.ndarray:
        x = self.rng.uniform(-self.world_scale, self.world_scale)
        y = self.rng.uniform(-self.world_scale, self.world_scale)
        z = self.rng.uniform(0.8, 2.0)
        return np.array([x, y, z], dtype=np.float32)

    def _make_targets(self) -> np.ndarray:
        if self.shared_target:
            t = self._make_shared_target()
            return np.repeat(t[None, :], repeats=self.num_drones, axis=0).astype(np.float32)

        targets = []
        for _ in range(self.num_drones):
            x = self.rng.uniform(-self.world_scale, self.world_scale)
            y = self.rng.uniform(-self.world_scale, self.world_scale)
            z = self.rng.uniform(0.8, 2.0)
            targets.append([x, y, z])
        return np.array(targets, dtype=np.float32)

    # -------- GUI-only visuals --------
    def _clear_debug_visuals(self) -> None:
        if self._p is None or self.render_mode != "human":
            self._target_body_ids = []
            self._target_text_ids = []
            self._lidar_line_ids = {}
            return

        p = self._p

        for bid in self._target_body_ids:
            self._call_draw(p.removeBody, bid)
        self._target_body_ids = []

        for tid in self._target_text_ids:
            self._call_draw(p.removeUserDebugItem, tid)
        self._target_text_ids = []

        for _, line_ids in list(self._lidar_line_ids.items()):
            for lid in line_ids:
                self._call_draw(p.removeUserDebugItem, lid)
        self._lidar_line_ids = {}

    def _spawn_target_markers(self) -> None:
        if self._p is None or self._targets is None or self.render_mode != "human":
            return
        p = self._p

        vis = self._call_draw(
            p.createVisualShape,
            p.GEOM_SPHERE,
            radius=0.10,
            rgbaColor=[0.1, 1.0, 0.1, 0.9],
        )
        if vis is None:
            return

        tgt = self._targets[0] if self.shared_target else None

        if self.shared_target and tgt is not None:
            body = self._call_draw(
                p.createMultiBody,
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vis,
                basePosition=[float(tgt[0]), float(tgt[1]), float(tgt[2])],
            )
            if body is not None:
                self._target_body_ids.append(int(body))

            tid = self._call_draw(
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
            body = self._call_draw(
                p.createMultiBody,
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vis,
                basePosition=[float(tgt_i[0]), float(tgt_i[1]), float(tgt_i[2])],
            )
            if body is not None:
                self._target_body_ids.append(int(body))

            tid = self._call_draw(
                p.addUserDebugText,
                f"T{i}",
                [float(tgt_i[0]), float(tgt_i[1]), float(tgt_i[2] + 0.15)],
                textColorRGB=[0.2, 1.0, 0.2],
                textSize=1.2,
                lifeTime=0,
            )
            if tid is not None:
                self._target_text_ids.append(int(tid))

    # -------- obstacles (physics client) --------
    def _spawn_obstacles(self) -> None:
        self._obstacle_ids = []
        self._obstacle_base_pos = None
        self._obstacle_phase = None

        if self.num_obstacles <= 0 or self._p is None:
            return

        p = self._p
        col = self._call_physics(p.createCollisionShape, p.GEOM_SPHERE, radius=self.obstacle_radius)
        vis = self._call_physics(
            p.createVisualShape, p.GEOM_SPHERE, radius=self.obstacle_radius, rgbaColor=[1, 0, 0, 1]
        )

        base_positions = []
        phases = []

        for _ in range(self.num_obstacles):
            x = self.rng.uniform(-self.world_scale, self.world_scale)
            y = self.rng.uniform(-self.world_scale, self.world_scale)
            z = self.rng.uniform(self.obstacle_height_range[0], self.obstacle_height_range[1])

            body = self._call_physics(
                p.createMultiBody,
                baseMass=0.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[x, y, z],
            )
            self._obstacle_ids.append(int(body))
            base_positions.append([x, y, z])
            phases.append(self.rng.uniform(0.0, 2.0 * np.pi))

        self._obstacle_base_pos = np.array(base_positions, dtype=np.float32)
        self._obstacle_phase = np.array(phases, dtype=np.float32)

    def _update_obstacles(self) -> None:
        if (
            (not self.obstacle_dynamic)
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

            dx = self.obstacle_drift_amp * np.cos(self.obstacle_drift_omega * t + ph)
            dy = self.obstacle_drift_amp * np.sin(self.obstacle_drift_omega * t + ph)

            pos = [float(base[0] + dx), float(base[1] + dy), float(base[2])]
            self._call_physics(p.resetBasePositionAndOrientation, oid, posObj=pos, ornObj=[0, 0, 0, 1])

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

    def _check_collision_any(self) -> bool:
        if self._p is None or len(self._obstacle_ids) == 0 or len(self._drone_ids) == 0:
            return False

        p = self._p
        thresh = self.obstacle_radius + self.collision_distance
        active_ids = (
            self._drone_ids[: self.active_drones] if len(self._drone_ids) >= self.active_drones else self._drone_ids
        )

        for did in active_ids:
            for oid in self._obstacle_ids:
                try:
                    pts = self._call_physics(p.getClosestPoints, bodyA=did, bodyB=oid, distance=thresh)
                    if pts and len(pts) > 0:
                        return True
                except Exception:
                    continue
        return False

    # ---------------- swarm helpers ----------------
    def _active_positions(self) -> np.ndarray:
        pos = np.zeros((self.active_drones, 3), dtype=np.float32)
        for i in range(self.active_drones):
            state = self._env._getDroneStateVector(i)
            pos[i] = np.array(state[0:3], dtype=np.float32)
        return pos

    def _swarm_centroid(self, positions: np.ndarray) -> np.ndarray:
        if positions.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)
        return positions.mean(axis=0).astype(np.float32)

    def _nearest_teammate_vector(self, i: int, positions: np.ndarray) -> np.ndarray:
        if positions.shape[0] <= 1:
            return np.zeros(3, dtype=np.float32)
        pi = positions[i]
        d = positions - pi[None, :]
        dist = np.linalg.norm(d, axis=1) + 1e-8
        dist[i] = 1e9
        j = int(np.argmin(dist))
        return (positions[j] - pi).astype(np.float32)

    def _pairwise_sep_stats(self, positions: np.ndarray) -> Tuple[float, float, bool, bool]:
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
                if d > self.swarm_max_sep:
                    too_far = True
                if d < self.swarm_min_sep:
                    too_close = True

        return max_d, min_d, too_far, too_close

    # ---------------- lidar ----------------
    def _precompute_lidar_dirs_body(self) -> np.ndarray:
        h_fov = np.deg2rad(max(1e-6, self.lidar_horizontal_fov_deg))
        v_fov = np.deg2rad(max(0.0, self.lidar_vertical_fov_deg))

        if self.lidar_horizontal_fov_deg >= 359.0:
            az = np.linspace(0.0, 2.0 * np.pi, num=self.lidar_num_rays, endpoint=False, dtype=np.float32)
        else:
            az = np.linspace(-0.5 * h_fov, 0.5 * h_fov, num=self.lidar_num_rays, endpoint=True, dtype=np.float32)

        if self.lidar_num_layers <= 1 or self.lidar_vertical_fov_deg <= 1e-6:
            el = np.array([0.0], dtype=np.float32)
        else:
            el = np.linspace(-0.5 * v_fov, 0.5 * v_fov, num=self.lidar_num_layers, endpoint=True, dtype=np.float32)

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
        ray_to = ray_from + dirs_world * self.lidar_range

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

        if self.lidar_noise_std > 0.0:
            hit_frac = np.clip(
                hit_frac + self.rng.normal(0.0, self.lidar_noise_std, size=hit_frac.shape).astype(np.float32),
                0.0,
                1.0,
            )

        # GUI debug drawing
        if self.render_mode == "human" and self.lidar_show_rays:
            old = self._lidar_line_ids.get(drone_idx, [])
            for lid in old:
                self._call_draw(p.removeUserDebugItem, lid)

            new_ids: List[int] = []
            for i in range(dirs_world.shape[0]):
                endp = hit_pos[i] if hit_frac[i] < 0.999 else ray_to[i]
                col = [1.0, 0.2, 0.2] if hit_frac[i] < 0.999 else [0.2, 0.6, 1.0]
                lid = self._call_draw(
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

    # ---------------- env API ----------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._step_count = 0
        self.agents = self.possible_agents[:]

        self._env.reset()
        self._init_pybullet_handles()
        self._refresh_drone_ids_from_pybullet()

        self._targets = self._make_targets()

        self._clear_debug_visuals()
        self._spawn_obstacles()
        self._spawn_target_markers()

        self._prev_dists = np.zeros((self.num_drones,), dtype=np.float32)
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

    def _build_obs_dict(self) -> Dict[str, np.ndarray]:
        obs_dict: Dict[str, np.ndarray] = {}

        if self._targets is None:
            self._targets = self._make_targets()

        positions = self._active_positions()
        centroid = self._swarm_centroid(positions)

        for i, a in enumerate(self.possible_agents):
            if i >= self.active_drones:
                obs_dict[a] = np.zeros((self._base_obs_dim + self._lidar_dim,), dtype=np.float32)
                continue

            state = self._env._getDroneStateVector(i)
            pos = np.array(state[0:3], dtype=np.float32)
            vel = np.array(state[10:13], dtype=np.float32)

            target = self._targets[i]
            to_target = (target - pos).astype(np.float32)

            to_obs = self._nearest_obstacle_vector(pos)
            to_centroid = (centroid - pos).astype(np.float32)
            to_nearest_teammate = self._nearest_teammate_vector(i, positions)

            base = np.concatenate([pos, vel, to_target, to_obs, to_centroid, to_nearest_teammate], axis=0).astype(
                np.float32
            )

            if self.lidar:
                scan = self._lidar_scan(i, pos)
                obs_dict[a] = np.concatenate([base, scan], axis=0).astype(np.float32)
            else:
                obs_dict[a] = base

        return obs_dict

    def step(self, actions: Dict[str, np.ndarray]):
        self._step_count += 1
        self._update_obstacles()

        # Convert actions -> VelocityAviary expected format [dir_x, dir_y, dir_z, speed]
        act = np.zeros((self.num_drones, 4), dtype=np.float32)

        for i, a in enumerate(self.possible_agents):
            if i >= self.active_drones:
                act[i] = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
                continue

            # default: hover-ish
            dir_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            speed = 0.0

            if a in actions:
                u = np.array(actions[a], dtype=np.float32).reshape(-1)
                u = np.clip(u[:4], -1.0, 1.0)

                v_dir = u[0:3].astype(np.float32)
                n = float(np.linalg.norm(v_dir))
                if n > 1e-6:
                    dir_vec = v_dir / n
                else:
                    dir_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)

                # map u[3] from [-1,1] -> [0,1]
                speed01 = float((u[3] + 1.0) * 0.5)
                speed = self.vel_scale * speed01

                # small constant vertical bias if you want
                dir_vec = dir_vec.copy()
                dir_vec[2] = float(dir_vec[2] + self.hover_bias_vz)

                # renormalize direction after bias
                nn = float(np.linalg.norm(dir_vec))
                if nn > 1e-6:
                    dir_vec = dir_vec / nn

            # takeoff override
            if self._step_count <= self.takeoff_steps:
                dir_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                speed = float(abs(self.takeoff_vz))

            act[i, 0:3] = dir_vec
            act[i, 3] = float(speed)

        self._env.step(act)

        obs_dict = self._build_obs_dict()

        reward_dict: Dict[str, float] = {}
        term_dict: Dict[str, bool] = {}
        trunc_dict: Dict[str, bool] = {}

        time_limit = (self._step_count >= self.max_steps)
        collided = self._check_collision_any()

        positions = self._active_positions()
        max_sep, min_sep, too_far, too_close = self._pairwise_sep_stats(positions)

        if self._prev_dists is None:
            self._prev_dists = np.zeros((self.num_drones,), dtype=np.float32)

        cohesion_pen = 0.0
        if positions.shape[0] > 1 and max_sep > self.swarm_max_sep:
            cohesion_pen = -self.cohesion_penalty_k * (max_sep - self.swarm_max_sep)

        sep_pen = 0.0
        if positions.shape[0] > 1 and min_sep < self.swarm_min_sep:
            sep_pen = -self.separation_penalty_k * (self.swarm_min_sep - min_sep)

        swarm_term = (too_far and self.terminate_on_cohesion_break) or (too_close and self.terminate_on_separation)

        # rewards + per-agent termination
        for i, a in enumerate(self.possible_agents):
            if i >= self.active_drones:
                reward_dict[a] = 0.0
                term_dict[a] = True
                trunc_dict[a] = bool(time_limit)
                continue

            pos = obs_dict[a][0:3]
            target = self._targets[i] if self._targets is not None else np.zeros(3, dtype=np.float32)

            dist = float(np.linalg.norm(target - pos))
            prev = float(self._prev_dists[i])
            self._prev_dists[i] = dist

            r_progress = self.progress_k * (prev - dist)
            r_goal_dist = -0.15 * dist          #WAS 0.05
            r_time = -self.time_penalty
            r_ctrl = -self.ctrl_penalty_k * float(np.linalg.norm(act[i]))
            r_near = -self.near_obstacle_penalty_k / max(float(np.linalg.norm(obs_dict[a][9:12])), 0.10)

            r = r_progress + r_goal_dist + r_time + r_ctrl + r_near
            r += float(cohesion_pen + sep_pen)

            reached = (dist < self.goal_radius)
            if reached:
                r += self.goal_bonus

            if collided:
                r -= self.collision_penalty

            reward_dict[a] = float(r)
            term_dict[a] = bool(collided or reached or swarm_term)
            trunc_dict[a] = bool(time_limit)

        # keep fixed agent list
        self.agents = self.possible_agents[:]

        # per-agent infos for callbacks
        info_dict: Dict[str, dict] = {}
        for i, a in enumerate(self.possible_agents):
            if i >= self.active_drones:
                info_dict[a] = {
                    "step": int(self._step_count),
                    "active": False,
                    "collided": bool(collided),
                    "reached": False,
                    "swarm_max_sep": float(max_sep),
                    "swarm_min_sep": float(min_sep),
                    "swarm_too_far": bool(too_far),
                    "swarm_too_close": bool(too_close),
                }
                continue

            pos = obs_dict[a][0:3]
            target = self._targets[i] if self._targets is not None else np.zeros(3, dtype=np.float32)
            dist = float(np.linalg.norm(target - pos))
            reached = (dist < self.goal_radius)

            info_dict[a] = {
                "step": int(self._step_count),
                "active": True,
                "collided": bool(collided),
                "reached": bool(reached),
                "swarm_max_sep": float(max_sep),
                "swarm_min_sep": float(min_sep),
                "swarm_too_far": bool(too_far),
                "swarm_too_close": bool(too_close),
            }

        return obs_dict, reward_dict, term_dict, trunc_dict, info_dict

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
