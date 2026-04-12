# -*- coding: utf-8 -*-
import numpy as np
from config.defaults import LIDAR_NUM_RAYS, LIDAR_NUM_LAYERS


class ExpertPolicy:
    """
    Reactive 3-D expert:
      1. Fly toward target.
      2. If obstacle is ahead, project target direction away from it.
      3. Use LiDAR to softly bias away from clutter.
      4. Avoid walls.
      5. Avoid teammates.
      6. Slow down smoothly near any danger.

    Action result: [dir_x, dir_y, dir_z, speed] in [-1, 1]
    """

    def __init__(
        self,
        avoid_radius: float = 3.5,  # if obstacle in radius -> react
        wall_margin: float = 2.5,  # if wall in margin -> react
        target_brake_dist: float = 1.5,  # if target in radius -> brake
        teammate_avoid_radius: float = 2.0,  # if teammate in radius -> react

        world_scale: float = 5.0,
        use_lidar: bool = True,

        speed_default: float = 0.75,
        speed_near_obs: float = 0.25,
        speed_near_target: float = 0.30,
        speed_near_teammate: float = 0.30,

        lidar_weight: float = 0.30,  # how strongly lidar influences final direction
        wall_weight: float = 0.55,  # how strongly wall avoidance influences final direction
        teammate_weight: float = 0.8,  # how strongly teammate avoidance influences final direction

        fly_over_gain: float = 0.30,  # obstacle directly ahead -> add  upward bias so the drone can go over it
    ):
        self.avoid_radius = avoid_radius
        self.wall_margin = wall_margin
        self.target_brake_dist = target_brake_dist
        self.teammate_avoid_radius = teammate_avoid_radius

        self.world_scale = world_scale
        self.use_lidar = use_lidar

        self.speed_default = speed_default
        self.speed_near_obs = speed_near_obs
        self.speed_near_target = speed_near_target
        self.speed_near_teammate = speed_near_teammate

        self.lidar_weight = lidar_weight
        self.wall_weight = wall_weight
        self.teammate_weight = teammate_weight

        self.fly_over_gain = fly_over_gain

    def action(self, obs: np.ndarray) -> np.ndarray:
        """
        Derive the expert action given the current observation.

        Returns: Action: [dir_x, dir_y, dir_z, speed] in [-1, 1]
        """
        obs = np.asarray(obs, dtype=np.float32)

        pos = obs[0:3]
        to_target = obs[6:9]
        to_obs = obs[9:12]
        to_teammate = obs[15:18]

        # get distance to target, nearest obstacle and nearest teammate
        dist_to_target = np.linalg.norm(to_target)
        dist_to_obs = np.linalg.norm(to_obs)
        dist_to_teammate = np.linalg.norm(to_teammate)

        # get direction to target
        dir_target = self.safe_unit(to_target)

        # 1) start from target direction
        steer_dir = dir_target.copy()

        # 2) modify direction to avoid nearest obstacle without orbiting
        steer_dir = self._steer_around_obstacle(obs, steer_dir)

        # 3) add soft lidar bias
        lidar_bias = self._lidar_bias(obs)

        # 4) add soft wall avoidance
        wall_bias = self._wall_avoidance(pos)

        # 5) add soft teammate avoidance
        teammate_bias = self._teammate_avoidance(obs, to_teammate, dist_to_teammate)

        action_dir = (
            steer_dir
            + self.lidar_weight * lidar_bias
            + self.wall_weight * wall_bias
            + self.teammate_weight * teammate_bias
        )
        action_dir = self.safe_unit(action_dir)

        # 5) speed control
        speed = self.speed_default

        # smooth slowdown inside obstacle influence radius
        if dist_to_obs < self.avoid_radius:
            alpha = np.clip(dist_to_obs / max(self.avoid_radius, 1e-6), 0.0, 1.0)
            # alpha=0 -> very close -> slow, alpha=1 -> far -> default
            speed_scale = self.speed_near_obs + alpha * (1.0 - self.speed_near_obs)
            speed *= speed_scale

        # slow near target
        if dist_to_target < self.target_brake_dist:
            alpha_t = np.clip(
                dist_to_target / max(self.target_brake_dist, 1e-6), 0.0, 1.0
            )
            target_scale = self.speed_near_target + alpha_t * (
                1.0 - self.speed_near_target
            )
            speed *= target_scale

        if dist_to_teammate < self.teammate_avoid_radius:
            alpha_m = np.clip(
                dist_to_teammate / max(self.teammate_avoid_radius, 1e-6), 0.0, 1.0
            )
            mate_scale = self.speed_near_teammate + alpha_m * (
                1.0 - self.speed_near_teammate
            )
            speed *= mate_scale

        speed_action = float(np.clip(speed * 2.0 - 1.0, -1.0, 1.0))

        return np.array(
            [
                float(np.clip(action_dir[0], -1.0, 1.0)),
                float(np.clip(action_dir[1], -1.0, 1.0)),
                float(np.clip(action_dir[2], -1.0, 1.0)),
                speed_action,
            ],
            dtype=np.float32,
        )

    def _steer_around_obstacle(
        self, obs: np.ndarray, dir_target: np.ndarray
    ) -> np.ndarray:
        """
        Avoid the nearest obstacle by removing the part of the target direction
        that points into it, instead of adding a big repulsive force. This avoids wide circular orbits.
        """
        to_obs = obs[9:12]
        dist_to_obs = np.linalg.norm(to_obs)

        # check whether (almost zero) distance / far enough
        if dist_to_obs <= 1e-6 or dist_to_obs >= self.avoid_radius:
            return dir_target

        # convert to unit vector
        obs_dir = self.safe_unit(to_obs)

        # how much the target direction points toward the obstacle
        forwardness = float(np.dot(dir_target, obs_dir))

        # obstacle not ahead -> keep target direction
        if forwardness <= 0.0:
            return dir_target

        # compute weight based on distance away from drone
        weight = np.clip(1.0 - dist_to_obs / max(self.avoid_radius, 1e-6), 0.0, 1.0)

        # Remove the component that points into obstacle
        tangential = dir_target - forwardness * obs_dir
        tangential_norm = np.linalg.norm(tangential)

        if tangential_norm < 1e-5:
            # obstacle is directly in front so pick horizontal direction
            side = np.array([-obs_dir[1], obs_dir[0], 0.0], dtype=np.float32)
            # check that it is still not small
            if np.linalg.norm(side) < 1e-5:
                side = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            tangential = self.safe_unit(side)
        else:
            tangential = tangential / tangential_norm

        # small upward bias when obstacle is strongly in front
        over = np.zeros(3, dtype=np.float32)
        if forwardness > 0.4:
            over[2] = self.fly_over_gain * weight

        new_direction_vec = (
            (1.0 - weight)
            * dir_target  # if weight small -> stay close to target direction
            + weight * tangential  # if weight large -> move more tangential
            + over
        )
        return self.safe_unit(new_direction_vec)

    def _lidar_bias(self, obs: np.ndarray) -> np.ndarray:
        """
        Soft repulsion with LiDAR.

        Returns a 3D array for the direction vector.
        """
        lidar_start = 18
        n_lidar = LIDAR_NUM_LAYERS * LIDAR_NUM_RAYS

        if not self.use_lidar:
            return np.zeros(3, dtype=np.float32)

        # check that LiDAR contains enough data
        if obs.shape[0] < lidar_start + n_lidar or n_lidar <= 0:
            return np.zeros(3, dtype=np.float32)

        lidar = obs[lidar_start : lidar_start + n_lidar]
        lidar = lidar.reshape(LIDAR_NUM_LAYERS, LIDAR_NUM_RAYS)

        # middle layer for horizontal steering
        mid = lidar[LIDAR_NUM_LAYERS // 2]
        angles = np.linspace(0.0, 2.0 * np.pi, LIDAR_NUM_RAYS, endpoint=False)
        # mid = 1 means no nearby obstacle, and vice versa
        proximity = np.clip(1.0 - mid, 0.0, 1.0)

        rx = -np.sum(proximity * np.cos(angles))
        ry = -np.sum(proximity * np.sin(angles))

        # top and bottom layers for vertical clearance
        top_prox = np.clip(1.0 - np.mean(lidar[0]), 0.0, 1.0)
        bot_prox = np.clip(1.0 - np.mean(lidar[-1]), 0.0, 1.0)
        rz = bot_prox - top_prox

        bias = np.array([rx, ry, rz], dtype=np.float32)

        # if lidar sees almost nothing, return zero
        if np.linalg.norm(bias) < 1e-3:
            return np.zeros(3, dtype=np.float32)

        # Scale by sensed clutter strength
        clutter = float(np.max(proximity))
        bias = self.safe_unit(bias) * clutter

        return self.clamp_norm(bias, 1.0)

    def _wall_avoidance(self, pos: np.ndarray) -> np.ndarray:
        """
        Soft repulsion from wall.

        Returns a 3D array for the direction vector.

        World:
          x in [-s, s]
          y in [-s, s]
          z in [0, s]
        """
        x, y, z = map(float, pos)
        s, m = self.world_scale, self.wall_margin
        avoid = np.zeros(3, dtype=np.float32)

        def wall_push(dist_to_wall: float, gain: float = 1.0) -> float:
            """
            If farther than wall margin -> no push
            If within wall margin -> push increases linearly as drone gets closer
            If at wall -> maximum push
            """
            if dist_to_wall >= m:
                return 0.0
            return gain * (1.0 - max(dist_to_wall, 0.0) / m)

        # x walls
        avoid[0] += wall_push(s - x, gain=1.0) * -1.0  # near +x wall -> push -x
        avoid[0] += wall_push(x + s, gain=1.0) * 1.0  # near -x wall -> push +x

        # y walls
        avoid[1] += wall_push(s - y, gain=1.0) * -1.0  # near +y wall -> push -y
        avoid[1] += wall_push(y + s, gain=1.0) * 1.0  # near -y wall -> push +y

        # z bounds
        avoid[2] += wall_push(s - z, gain=0.8) * -1.0  # near ceiling -> push down
        avoid[2] += wall_push(z, gain=1.0) * 1.0  # near floor   -> push up

        return self.clamp_norm(avoid, 1.0)

    def _teammate_avoidance(
        self, obs: np.ndarray, teammate_pos: np.ndarray, distance_to_teammate: float
    ) -> np.ndarray:
        """
        Soft repulsion from nearest teammate.

        Returns a 3D array for the direction vector.
        """
        # check for almost zero distance / teammate far enough away
        if (
            distance_to_teammate <= 1e-6
            or distance_to_teammate >= self.teammate_avoid_radius
        ):
            return np.zeros(3, dtype=np.float32)

        # compute unit vector from teammate
        away = -self.safe_unit(teammate_pos)

        # stronger weight applied when closer
        weight = 1.0 - distance_to_teammate / max(self.teammate_avoid_radius, 1e-6)

        # preference to separate horizontally first (avoids just flying up)
        away = away.copy()
        away[2] *= 0.5

        # scale by closeness and clamp norm
        return self.clamp_norm(weight * away, 1.0)

    def safe_unit(self, v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Normalise a vector; return zero-vector if it's near-zero."""
        n = np.linalg.norm(v)
        return v / (n + eps) if n > eps else np.zeros_like(v)

    def clamp_norm(
        self, v: np.ndarray, max_norm: float = 1.0, eps: float = 1e-8
    ) -> np.ndarray:
        """Clamp vector magnitude without destroying small magnitudes."""
        n = np.linalg.norm(v)
        if n <= max_norm:
            return v
        return v / (n + eps) * max_norm
