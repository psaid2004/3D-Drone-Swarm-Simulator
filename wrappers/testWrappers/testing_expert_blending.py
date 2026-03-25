import time
import numpy as np
from wrappers.env_3d import DroneSwarm3D
from config.defaults import LIDAR_NUM_RAYS, LIDAR_NUM_LAYERS


def safe_unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalise a vector; return zero-vector if it's near-zero."""
    n = np.linalg.norm(v)
    return v / (n + eps) if n > eps else np.zeros_like(v)


def clamp_norm(v: np.ndarray, max_norm: float = 1.0, eps: float = 1e-8) -> np.ndarray:
    """Clamp vector magnitude without destroying small magnitudes."""
    n = np.linalg.norm(v)
    if n <= max_norm:
        return v
    return v / (n + eps) * max_norm


class ExpertPolicy:
    """
    Reactive 3-D expert:
      1. Fly toward target.
      2. If obstacle is ahead, project target direction away from it.
      3. Add a small LiDAR-based steering bias.
      4. Add wall avoidance.
      5. Slow down smoothly near danger.

    Observation layout:
      [0:3]   pos
      [3:6]   vel
      [6:9]   to_target
      [9:12]  to_nearest_obs
      [12:15] to_centroid
      [15:18] to_nearest_teammate
      [18:]   lidar hit fractions  (LIDAR_NUM_LAYERS × LIDAR_NUM_RAYS)

    Action: [dir_x, dir_y, dir_z, speed] in [-1, 1]
    """

    def __init__(
        self,
        avoid_radius: float = 3.0,
        wall_margin: float = 2.0,
        world_scale: float = 5.0,
        use_lidar: bool = True,
        speed_default: float = 0.75,
        speed_near_obs: float = 0.25,
        speed_near_target: float = 0.30,
        speed_near_teammate: float = 0.35,
        target_brake_dist: float = 1.5,
        obstacle_dir_weight: float = 1.0,
        lidar_weight: float = 0.35,
        wall_weight: float = 0.55,
        teammate_avoid_radius: float = 2.0,
        teammate_weight: float = 0.8,
        fly_over_gain: float = 0.35,
    ):
        self.avoid_radius = avoid_radius
        self.wall_margin = wall_margin
        self.world_scale = world_scale
        self.use_lidar = use_lidar

        self.speed_default = speed_default
        self.speed_near_obs = speed_near_obs
        self.speed_near_target = speed_near_target
        self.speed_near_teammate = speed_near_teammate
        self.target_brake_dist = target_brake_dist

        self.obstacle_dir_weight = obstacle_dir_weight
        self.lidar_weight = lidar_weight
        self.wall_weight = wall_weight
        self.teammate_avoid_radius = teammate_avoid_radius
        self.teammate_weight = teammate_weight
        self.fly_over_gain = fly_over_gain


    def action(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)

        pos = obs[0:3]
        to_target = obs[6:9]
        to_obs = obs[9:12]

        dist_to_target = np.linalg.norm(to_target)
        dist_to_obs = np.linalg.norm(to_obs)

        dir_target = safe_unit(to_target)

        to_mate = obs[15:18]
        dist_to_mate = np.linalg.norm(to_mate)
        teammate_bias = self._teammate_avoidance(obs)

        # 1) start from target direction
        steer_dir = dir_target.copy()

        # 2) modify direction to avoid nearest obstacle without orbiting
        steer_dir = self._steer_around_obstacle(obs, steer_dir)

        # 3) add soft lidar bias
        lidar_bias = self._lidar_bias(obs)

        # 4) add soft wall avoidance
        wall_bias = self._wall_avoidance(pos)

        action_dir = (
                steer_dir
                + self.lidar_weight * lidar_bias
                + self.wall_weight * wall_bias
                + self.teammate_weight * teammate_bias
        )
        action_dir = safe_unit(action_dir)

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
            alpha_t = np.clip(dist_to_target / max(self.target_brake_dist, 1e-6), 0.0, 1.0)
            target_scale = self.speed_near_target + alpha_t * (1.0 - self.speed_near_target)
            speed *= target_scale

        if dist_to_mate < self.teammate_avoid_radius:
            alpha_m = np.clip(dist_to_mate / max(self.teammate_avoid_radius, 1e-6), 0.0, 1.0)
            mate_scale = self.speed_near_teammate + alpha_m * (1.0 - self.speed_near_teammate)
            speed *= mate_scale

        speed_action = float(np.clip(speed * 2.0 - 1.0, -1.0, 1.0))

        return np.array([
            float(np.clip(action_dir[0], -1.0, 1.0)),
            float(np.clip(action_dir[1], -1.0, 1.0)),
            float(np.clip(action_dir[2], -1.0, 1.0)),
            speed_action,
        ], dtype=np.float32)


    def _steer_around_obstacle(self, obs: np.ndarray, dir_target: np.ndarray) -> np.ndarray:
        """
        Avoid the nearest obstacle by removing the part of the target direction
        that points into it, instead of adding a big repulsive force.
        This avoids wide circular orbits.
        """
        to_obs = obs[9:12]
        dist_to_obs = np.linalg.norm(to_obs)

        if dist_to_obs <= 1e-6 or dist_to_obs >= self.avoid_radius:
            return dir_target

        obs_dir = safe_unit(to_obs)

        # How much the target direction points toward the obstacle
        forwardness = float(np.dot(dir_target, obs_dir))

        # Obstacle not ahead => keep target direction
        if forwardness <= 0.0:
            return dir_target

        weight = np.clip(1.0 - dist_to_obs / max(self.avoid_radius, 1e-6), 0.0, 1.0)

        # Remove the component that points into obstacle
        tangential = dir_target - forwardness * obs_dir
        tangential_norm = np.linalg.norm(tangential)

        if tangential_norm < 1e-5:
            # Nearly head-on: pick a horizontal sidestep
            side = np.array([-obs_dir[1], obs_dir[0], 0.0], dtype=np.float32)
            if np.linalg.norm(side) < 1e-5:
                side = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            tangential = safe_unit(side)
        else:
            tangential = tangential / tangential_norm

        # Small upward bias when obstacle is strongly ahead
        over = np.zeros(3, dtype=np.float32)
        if forwardness > 0.4:
            over[2] = self.fly_over_gain * weight

        corrected = (
            (1.0 - weight) * dir_target
            + weight * tangential
            + over
        )
        return safe_unit(corrected)


    def _lidar_bias(self, obs: np.ndarray) -> np.ndarray:
        """
        LiDAR creates a soft steering bias away from clutter.
        This is intentionally weak to avoid oscillation.
        """
        lidar_start = 18
        n_lidar = LIDAR_NUM_LAYERS * LIDAR_NUM_RAYS

        if not self.use_lidar:
            return np.zeros(3, dtype=np.float32)

        if obs.shape[0] < lidar_start + n_lidar or n_lidar <= 0:
            return np.zeros(3, dtype=np.float32)

        lidar = obs[lidar_start: lidar_start + n_lidar]
        lidar = lidar.reshape(LIDAR_NUM_LAYERS, LIDAR_NUM_RAYS)

        # middle layer for horizontal steering
        mid = lidar[LIDAR_NUM_LAYERS // 2]
        angles = np.linspace(0.0, 2.0 * np.pi, LIDAR_NUM_RAYS, endpoint=False)
        proximity = np.clip(1.0 - mid, 0.0, 1.0)

        rx = -np.sum(proximity * np.cos(angles))
        ry = -np.sum(proximity * np.sin(angles))

        # top and bottom layers for vertical clearance
        top_prox = np.clip(1.0 - np.mean(lidar[0]), 0.0, 1.0)
        bot_prox = np.clip(1.0 - np.mean(lidar[-1]), 0.0, 1.0)
        rz = bot_prox - top_prox

        bias = np.array([rx, ry, rz], dtype=np.float32)

        # If lidar sees almost nothing, return zero
        if np.linalg.norm(bias) < 1e-3:
            return np.zeros(3, dtype=np.float32)

        # Scale by sensed clutter strength
        clutter = float(np.max(proximity))
        bias = safe_unit(bias) * clutter
        return clamp_norm(bias, 1.0)


    def _wall_avoidance(self, pos: np.ndarray) -> np.ndarray:
        """
        World:
          x in [-s, s]
          y in [-s, s]
          z in [0, s]
        """
        x, y, z = map(float, pos)
        s, m = self.world_scale, self.wall_margin
        avoid = np.zeros(3, dtype=np.float32)

        def wall_push(dist_to_wall: float, gain: float = 1.0) -> float:
            if dist_to_wall >= m:
                return 0.0
            return gain * (1.0 - max(dist_to_wall, 0.0) / m)

        # x walls
        avoid[0] += wall_push(s - x, gain=1.0) * -1.0   # near +x wall -> push -x
        avoid[0] += wall_push(x + s, gain=1.0) *  1.0   # near -x wall -> push +x

        # y walls
        avoid[1] += wall_push(s - y, gain=1.0) * -1.0   # near +y wall -> push -y
        avoid[1] += wall_push(y + s, gain=1.0) *  1.0   # near -y wall -> push +y

        # z bounds
        avoid[2] += wall_push(s - z, gain=0.8) * -1.0   # near ceiling -> push down
        avoid[2] += wall_push(z,     gain=1.0) *  1.0   # near floor   -> push up

        return clamp_norm(avoid, 1.0)

    def _teammate_avoidance(self, obs: np.ndarray) -> np.ndarray:
        """
        Soft repulsion from nearest teammate.
        Assumes obs[15:18] = nearest_teammate - pos
        """
        to_mate = obs[15:18]
        dist = np.linalg.norm(to_mate)

        if dist <= 1e-6 or dist >= self.teammate_avoid_radius:
            return np.zeros(3, dtype=np.float32)

        # Repel away from teammate
        away = -safe_unit(to_mate)

        # Stronger when closer
        weight = 1.0 - dist / max(self.teammate_avoid_radius, 1e-6)

        # Slight preference to separate horizontally first
        away = away.copy()
        away[2] *= 0.5

        return clamp_norm(weight * away, 1.0)


def main():
    n_drones = 10
    n_steps = 8000
    use_lidar = True
    world_scale = 3.0

    env = DroneSwarm3D(
        world_scale=world_scale,
        num_drones_per_swarm=n_drones,
        num_static_obstacles=40,
        num_dynamic_obstacles=4,
        max_steps=n_steps,
        num_swarms=1,
        lidar=use_lidar,
        # render_mode="human",
    )

    expert = ExpertPolicy(
        avoid_radius=3.5,
        wall_margin=2.5,
        world_scale=world_scale,
        use_lidar=use_lidar,
        speed_default=0.75,
        speed_near_obs=0.20,
        speed_near_target=0.30,
        target_brake_dist=1.5,
        obstacle_dir_weight=1.0,
        lidar_weight=0.30,
        wall_weight=0.55,
        fly_over_gain=0.30,
    )

    obs_dict, info = env.reset()

    print("Initial observations:")
    for drone, obs in obs_dict.items():
        print(f"  {drone}: pos={obs[:3]}")

    for step in range(n_steps):
        action_dict = {
            drone: expert.action(obs)
            for drone, obs in obs_dict.items()
        }

        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)

        if step % 50 == 0:
            print(f"\n===== STEP {step} =====")
            for d in obs_dict:
                pos = obs_dict[d][:3]
                r = reward_dict[d]
                collided = info_dict[d].get("collided", False)
                reached = info_dict[d].get("reached", False)
                wall = info_dict[d].get("wall_collision", False)
                obs_hit = info_dict[d].get("obstacle_collision", False)
                drone_hit = info_dict[d].get("drone_collision", False)
                print(
                    f"  {d:8s} | "
                    f"pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:6.2f}) | "
                    f"reward={r:7.2f} | "
                    f"collided={str(collided):5s} | "
                    f"reached={str(reached):5s} | "
                    f"wall={wall} obs={obs_hit} drone={drone_hit}"
                )

        if all(term_dict.values()):
            print(f"Episode ended at step {step}")
            for d in obs_dict:
                pos = obs_dict[d][:3]
                r = reward_dict[d]
                collided = info_dict[d].get("collided", False)
                reached = info_dict[d].get("reached", False)
                wall = info_dict[d].get("wall_collision", False)
                obs_hit = info_dict[d].get("obstacle_collision", False)
                drone_hit = info_dict[d].get("drone_collision", False)
                print(
                    f"  {d:8s} | "
                    f"pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:6.2f}) | "
                    f"reward={r:7.2f} | "
                    f"collided={str(collided):5s} | "
                    f"reached={str(reached):5s} | "
                    f"wall={wall} obs={obs_hit} drone={drone_hit}"
                )
            break

        env.render()
        time.sleep(1 / 60)

    input("Press Enter to close...")
    env.close()


if __name__ == "__main__":
    main()