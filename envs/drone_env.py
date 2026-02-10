import time

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils.seeding import np_random

# -------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# -------------------------------------------------------------------------
DT = 0.1  # 10 Hz physics step
ARENA_SIZE = 100.0  # metres (square arena: −50..+50 in both axes)

# Kinematic limits --------------------------------------------------------
MAX_V = 5  # m/s
MAX_A = 2.0  # m/s^2

# Drone / obstacle geometry ----------------------------------------------
R_COLL_MIN, R_COLL_MAX = 0.5, 0.5  # drones are discs with this radius range
OBS_MIN_R, OBS_MAX_R = 1.0, 3.0  # static obstacle radii
GOAL_WALL_MARGIN = 3.5  # Minimum distance between the wall and the goal
SUCCESS_DIST = 1.5  # Distance at which the drone is assumed to land goal
MIN_GOAL_DIST = 15.0
SPAWN_MARGIN = 6.0  # Minimum Distance Between agent spawn and another agent/obstacle

# Reward Functions Tunables

SAFE_DIST = 11.0
DANGEROUS_DIST = 8.0
FUNNEL_R = 15.0
BRAKE_R = 9.0
ALIGN_R = 15.0
MIN_PUSH_SPEED = 0.5

# core coefficients
PROG_COEF = 4.5
STEP_COST = 0.01
ACTION_PEN = 0.005
ALIGN_COEF = 0.5
BRAKE_COEF = 2.0
REP_COEF = 0.5

# Terminal Bonuses
COLLISION_PEN = 30.0
REACH_BONUS = 150.0
TRUNC_PEN = 300.0
ZERO_SPEED_PEN = 1.0

# dodging bonuses
AWAY_DRONE_COEF = 2.5
AWAY_OBS_COEF = 1.5

eps = 1e-8


# -------------------------------------------------------------------------

def make_palette(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = []
    for h in hues:
        col = pygame.Color(0)
        col.hsva = (h * 360, 80, 100, 100)
        colors.append((col.r, col.g, col.b))
    return colors


class DroneWorldEnv(gym.Env):
    """
    N‑agent 2‑D discs with independent goals.
    Single global step updates **all** drones; observations are returned
    as a list in fixed agent‑index order 0..N‑1.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, num_drones: int = 4, num_obstacles: int = 5, max_steps: int = 1000, lidar=False,
                 render_mode=None, max_v=MAX_V, max_a=MAX_A,
                 spawn_area=ARENA_SIZE):
        super().__init__()
        self.seed = None
        self.steps = 0
        self.n = num_drones
        self.dead = np.zeros(self.n, dtype=bool)
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.lidar = lidar
        self._ep_done = False
        self.wall_hits = 0
        self.max_v = max_v
        self.max_a = max_a
        self.obs_dim = 16 + (24 if self.lidar else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-max_a, max_a, shape=(2,), dtype=np.float32)
        # Visual -------------------------------------------------------------
        self.render_mode = render_mode or "human"
        self.palette = make_palette(self.n)
        self.np_random, _ = np_random(None)
        self._status = np.zeros(self.n, dtype=np.int8)
        self.prev_goal_dist = None
        self.last_reward = np.zeros(self.n, dtype=np.float32)
        self.spawn_area = spawn_area

    # ---------- Episode control ------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, self.seed = np_random(seed)
        self.steps = 0
        self.collision_count = 0
        self.reached_total = 0
        self.dead[:] = False
        self.wall_hits = 0
        self.time_since_reached = np.zeros(self.n, dtype=np.int32)
        self.reached_arr = np.zeros(self.n, dtype=bool)
        self.radii = self.np_random.uniform(R_COLL_MIN, R_COLL_MAX, size=self.n).astype(np.float32)
        self._spawn_obstacles()
        self._spawn_drones_and_goals()

        # ---------- Book‑keeping ----------
        self._ep_done = False
        self._status = np.zeros(self.n, dtype=np.int8)
        # distance to goal at *start* of step – required for potential term
        self.prev_goal_dist = np.linalg.norm(self.goal - self.pos, axis=1)
        self.last_reward = np.zeros(self.n, dtype=np.float32)

        info = {f"drone_{j}": {} for j in range(self.n)}
        return self._get_obs(), info

    # ------------------- Spawning Controls -------------------------------------------
    def _spawn_obstacles(self):
        self.obs_pos = np.zeros((self.num_obstacles, 2), dtype=np.float32)
        self.obs_rad = np.zeros(self.num_obstacles, dtype=np.float32)
        for k in range(self.num_obstacles):
            placed = False
            while not placed:
                centre = self.np_random.uniform(-self.spawn_area / 2, self.spawn_area / 2, size=2)
                rad = self.np_random.uniform(OBS_MIN_R, OBS_MAX_R)
                if k > 0:
                    d_other = np.linalg.norm(self.obs_pos[:k] - centre, axis=1)
                    if np.any(d_other < (self.obs_rad[:k] + rad) + 0.2):
                        continue
                self.obs_pos[k] = centre
                self.obs_rad[k] = rad
                placed = True

    def sample_in_box(self):
        (xmin, xmax), (ymin, ymax) = self.spawn_area
        x = self.np_random.uniform(xmin, xmax)
        y = self.np_random.uniform(ymin, ymax)
        return np.array([x, y], dtype=np.float32)

    def _spawn_drones_and_goals(self):
        self.pos = np.zeros((self.n, 2), dtype=np.float32)
        half = self.spawn_area / 2
        # ---- 1) spawn drone positions -----------------------------------
        for drone in range(self.n):
            placed = False
            r = self.radii[drone]
            while not placed:
                candidate = self.np_random.uniform(-half + r, half - r, size=2)
                if drone > 0:
                    d_drone = np.linalg.norm(self.pos[:drone] - candidate, axis=1)
                    if np.any(d_drone < (self.radii[:drone] + r + SPAWN_MARGIN)):
                        continue
                if self.num_obstacles > 0:
                    d_obs = np.linalg.norm(self.obs_pos - candidate, axis=1)
                    if np.any(d_obs < (self.obs_rad + r + SPAWN_MARGIN)):
                        continue
                self.pos[drone] = candidate
                placed = True
        self.vel = np.zeros((self.n, 2), dtype=np.float32)
        self.goal = np.zeros_like(self.pos)
        for drone in range(self.n):
            self.goal[drone] = self._sample_goal_for_agent(drone)

        # ------------------------------------------------------------------

    def _sample_goal_for_agent(self, idx: int) -> np.ndarray:
        """
        1) Try mirrored goal = -pos.
        2) If too close, or invalid, fall back to uniform sampling,
           but always preserve a minimum distance.
        """
        half = self.spawn_area / 2
        r = self.radii[idx]
        pos_i = self.pos[idx]

        # 1) mirrored candidate
        mirror = -pos_i.copy()
        if np.linalg.norm(mirror - pos_i) > MIN_GOAL_DIST and self._goal_is_valid(mirror, r, idx):
            return mirror

        # 2) uniform sampling with min‐dist rejection
        for _ in range(1000):
            cand = self.np_random.uniform(-half + r, half - r, size=2)
            if (self._goal_is_valid(cand, r, idx)
                    and np.linalg.norm(cand - pos_i) > MIN_GOAL_DIST):
                return cand

        # 3) fallback (unlikely)
        return pos_i + np.array([MIN_GOAL_DIST, 0.0], dtype=np.float32)

    def _goal_is_valid(self, goal: np.ndarray, r: float, id: int) -> bool:
        half = self.spawn_area / 2
        margin = GOAL_WALL_MARGIN + r  # margin for **centre** + drone radius

        # ---- keep the goal disc inside the arena -------------------------
        if (goal[0] < -half + margin) or (goal[0] > half - margin):
            return False
        if (goal[1] < -half + margin) or (goal[1] > half - margin):
            return False

        # ---- obstacle clearance (unchanged) ------------------------------
        if self.num_obstacles > 0:
            d_obs = np.linalg.norm(self.obs_pos - goal, axis=1)
            if np.any(d_obs < (self.obs_rad + r + 4.0)):
                return False

        if self.n > 1:
            existing_goals = self.goal[:id]
            if existing_goals.shape[0] > 0:
                d_goals = np.linalg.norm(existing_goals - goal, axis=1)
                if np.any(d_goals < (2.0 * r + SPAWN_MARGIN)):
                    return False

        return True

    # ---------- Step Computing --------------------------------------------------------

    def step(self, actions):
        """
        Parameters
        ----------
        actions : ndarray, shape (n, 2)
            Desired acceleration for each drone (ax, ay), clipped to MAX_A.

        Returns
        -------
        obs : ndarray (n, obs_dim)
        reward : ndarray (n,)
        done   : ndarray (n,)  – True when agent terminates
        trunc  : ndarray (n,)  – True when env truncates (time-limit)
        info   : dict(agent_id -> dict)
        """
        # Store previous distance‑to‑goal for potential shaping ------------
        prev_goal_dist = self.prev_goal_dist
        for i in range(self.n):
            if self.reached_arr[i]:  # if it has ever parked before
                actions[i, :] = 0.0
        # 1) Physics -------------------------------------------------------
        self._apply_physics(actions)

        # 2) Boundary collisions ------------------------------------------
        wall_hit = self._handle_walls()

        # 3) Terminal checks ----------------------------------------------
        reached = self._check_reached()
        # self._freeze_on_goal(reached)
        coll = self._check_collisions(wall_hit)
        # self._freeze_on_collision(coll)
        trunc = self._check_time_limit()
        # done = reached | coll
        # 4) Reward --------------------------------------------------------
        reward = self._compute_reward(actions, reached, coll, wall_hit, prev_goal_dist, trunc)
        self.last_reward = reward.copy()
        # 5) House‑keeping -------------------------------------------------
        self.prev_goal_dist = np.linalg.norm(self.goal - self.pos, axis=1)
        self._update_status(reached, coll, trunc)
        # new_collisions = coll & (~self.dead)
        self.dead |= coll
        self.collision_count += int(coll.sum())
        new_reached = reached & (~self.reached_arr)  # boolean AND-NOT
        self.reached_arr |= reached  # update memory
        self.reached_total += int(new_reached.sum())
        self.wall_hits += int(wall_hit.sum())
        done = self.reached_arr | trunc
        info = {}
        for i in range(self.n):
            info[f"drone_{i}"] = {
                "wall_hit": bool(wall_hit[i]),
                "collided": bool(coll[i]),
                "reached": bool(reached[i]),
            }

        # print(done.all() or trunc.any())
        # print(f'Done : {done}')
        # print(f'Tr : {trunc}')
        if done.all() or trunc.any():
            info["collision_count"] = int(self.collision_count)
            info["goals_reached"] = int(self.reached_total)
            info["wall_hits"] = int(self.wall_hits)
        else:
            info["collision_count"] = 0
            info["goals_reached"] = 0
        return self._get_obs(), reward, done, trunc, info

    # ---------- helpers --------------------------------------------------------

    def _get_obs(self):
        out = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        max_dist = ARENA_SIZE * np.sqrt(2)
        for i in range(self.n):
            # 1) own velocity
            v_norm = self.vel[i] / self.max_v  # 2 dims

            # 2) goal direction & distance
            g_vec = self.goal[i] - self.pos[i]
            g_dir = g_vec / (np.linalg.norm(g_vec) + 1e-8)
            g_dist = np.linalg.norm(g_vec) / max_dist

            # 3) wall margins (still in [0,1])
            half = self.spawn_area / 2
            wx = (half - abs(self.pos[i, 0]) - self.radii[i]) / half
            wy = (half - abs(self.pos[i, 1]) - self.radii[i]) / half
            wall = np.array([wx, wy], dtype=np.float32)  # 2 dims
            g = self.goal[i] - self.pos[i]
            dist = np.linalg.norm(g) + 1e-8
            ghat = g / dist
            vdir = self.vel[i] / (np.linalg.norm(self.vel[i]) + 1e-8)
            heading_cos = np.dot(vdir, ghat)
            # 4) nearest obstacle vector & distance
            if self.num_obstacles > 0:
                obs_dist_all = np.linalg.norm(self.obs_pos - self.pos[i], axis=1)
                j = np.argmin(obs_dist_all)
                rel_obs = (self.obs_pos[j] - self.pos[i]) / max_dist  # 2
                d_obs = (obs_dist_all[j] - self.radii[i] - self.obs_rad[j]) / max_dist  # 1
            else:
                rel_obs = np.zeros(2, dtype=np.float32)
                d_obs = 1.0

            # 5) nearest drone vector & relative velocity
            if self.n > 1:
                drone_dists = np.linalg.norm(self.pos - self.pos[i], axis=1)
                drone_dists[i] = np.inf
                k = np.argmin(drone_dists)
                rel_dr = (self.pos[k] - self.pos[i]) / max_dist  # 2
                dv_dr = (self.vel[k] - self.vel[i]) / self.max_v  # 2
                d_drone = (drone_dists[k] - self.radii[i] - self.radii[k]) / max_dist  # 1
            else:
                rel_dr, dv_dr = np.zeros(2), np.zeros(2)
                d_drone = 1.0
            obs_i = np.concatenate([
                v_norm,  # 2
                g_dir, [g_dist],  # 3
                wall,  # 2
                rel_obs, [d_obs],  # 3
                rel_dr, dv_dr,  # 4
                [d_drone],
                [heading_cos]  # 1
            ])
            if self.lidar:
                lidar = self._get_lidar(self.pos[i], i)
                obs_i = np.concatenate([obs_i, lidar])
            out[i] = obs_i
        return out

    def _get_lidar(self, pos_i, drone_i, n_beams=24, max_range=20.0):
        # Input validation
        assert n_beams > 0 and max_range > 0

        eps = 1e-6  # precision constant
        angles = np.linspace(0, 2 * np.pi, n_beams, endpoint=False)
        dists = np.full(n_beams, max_range, dtype=np.float32)
        half = ARENA_SIZE / 2.0

        for j, angle in enumerate(angles):
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

            # 1) Wall intersections
            for dim in (0, 1):
                for boundary in (half, -half):
                    denominator = direction[dim]
                    if abs(denominator) < eps:
                        continue  # parallel to wall
                    t = (boundary - pos_i[dim]) / denominator
                    if not (eps < t < dists[j] - eps):
                        continue
                    other = 1 - dim
                    coord = pos_i[other] + t * direction[other]
                    if -half - eps <= coord <= half + eps:
                        dists[j] = t

            # 2) Obstacle intersections
            if self.num_obstacles > 0:
                rels = self.obs_pos - pos_i
                ts = rels.dot(direction)
                mask = (ts > eps) & (ts < dists[j] - eps)
                perp = np.linalg.norm(rels - ts[:, None] * direction, axis=1)
                hit = mask & (perp < self.obs_rad - eps)
                if np.any(hit):
                    ts_hit = ts[hit]
                    perps = perp[hit]
                    rads = self.obs_rad[hit]
                    deltas = np.sqrt(np.maximum(0, rads ** 2 - perps ** 2))
                    # Ensure distance is non-negative by clamping at 0
                    dists[j] = np.minimum(dists[j], np.maximum(0, np.min(ts_hit - deltas)))

            # 3) Drone intersections
            rels = self.pos - pos_i
            ts = rels.dot(direction)
            mask = (ts > eps) & (ts < dists[j] - eps)
            for k in range(self.n):
                if k == drone_i or not mask[k]:
                    continue
                perp = np.linalg.norm(rels[k] - ts[k] * direction)
                rad_sum = self.radii[drone_i] + self.radii[k]
                if perp < rad_sum - eps:
                    delta = np.sqrt(max(0, rad_sum ** 2 - perp ** 2))
                    dists[j] = min(dists[j], max(0, ts[k] - delta))

        return np.clip(dists, 0.0, max_range) / max_range

    def _apply_physics(self, actions):
        # 1) standard accel + velocity cap
        a = np.clip(actions, -self.max_a, self.max_a)
        self.vel += a * DT
        speed = np.linalg.norm(self.vel, axis=1)
        over = speed > self.max_v
        if over.any():
            self.vel[over] *= (self.max_v / speed[over])[:, None]

        # 2) near‐goal damping
        near_goal = np.linalg.norm(self.goal - self.pos, axis=1) < 2.0
        self.vel[near_goal] *= 0.9

        # 3) tentative move
        new_pos = self.pos + self.vel * DT

        # 4) obstacle bounce‐back
        for i in range(self.n):
            for k in range(self.num_obstacles):
                # vector from obstacle center to drone
                rel = new_pos[i] - self.obs_pos[k]
                dist = np.linalg.norm(rel)
                min_dist = self.radii[i] + self.obs_rad[k]
                if dist < min_dist:
                    # a) compute contact normal (unit)
                    normal = rel / (dist + 1e-8)
                    # b) reflect velocity with a little extra bounce
                    v = self.vel[i]
                    v_n = np.dot(v, normal)
                    # reverse and amplify the normal component
                    v_reflected = v - (1.5 * v_n) * normal - v_n * normal
                    self.vel[i] = v_reflected
                    # c) nudge the drone just outside the obstacle
                    new_pos[i] = self.obs_pos[k] + normal * (min_dist + 1e-3)
                    # (optional) record this collision in info for your callback
                    # info[i]["collided_obs"] = True
                    break

        # 5) commit the (possibly corrected) positions
        self.pos = new_pos

    def _check_reached(self):
        curr_dist = np.linalg.norm(self.goal - self.pos, axis=1)
        close = curr_dist < SUCCESS_DIST

        # Only compute velocity checks for drones that are close to the goal
        valid = close  # Start with all close drones
        radial_vel = np.zeros_like(curr_dist)
        speed = np.zeros_like(curr_dist)

        if np.any(close):
            # Compute radial velocity for close drones
            r_vec = self.goal - self.pos
            dists = curr_dist[:, None] + 1e-8
            r_hat = r_vec / dists
            radial_vel = np.sum(self.vel * r_hat, axis=1)

            # Compute speed for close drones
            speed = np.linalg.norm(self.vel, axis=1)

        # Strict criteria for success:
        # 1. Must be within success distance
        # 2. Radial velocity must be small (both approaching and receding)
        # 3. Overall speed must be low
        return close & (speed < 0.7)

    def _freeze_on_goal(self, reached: np.ndarray):
        for i in np.where(reached)[0]:
            self.pos[i] = self.goal[i].copy()
            self.vel[i] = np.zeros(2, dtype=np.float32)

    def _freeze_on_collision(self, coll: np.ndarray):
        """
        For every drone that collided this step, zero out its velocity
        and leave its position unchanged so it “freezes in place.”
        """
        for i in np.where(coll)[0]:
            # pos stays where it collided
            # simply zero its velocity
            self.vel[i] = np.zeros_like(self.vel[i])

    def _check_collisions(self, wall_hit: np.ndarray):
        coll = np.zeros(self.n, dtype=bool)
        collision_buffer = 0.1  # Safety margin

        if self.n > 1:
            dmat = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=2)
            rsum = self.radii[:, None] + self.radii[None, :] + collision_buffer
            np.fill_diagonal(dmat, np.inf)  # ignore self
            coll |= (dmat < rsum).any(axis=1)

        if self.num_obstacles > 0:
            d_obs = np.linalg.norm(self.pos[:, None, :] - self.obs_pos[None, :, :], axis=2)
            rsum = self.radii[:, None] + self.obs_rad[None, :] + collision_buffer
            coll |= (d_obs < rsum).any(axis=1)

        return coll

    def _check_time_limit(self):
        self.steps += 1
        trunc = np.zeros(self.n, dtype=bool)
        if self.steps >= self.max_steps:
            trunc[:] = True
        return trunc

    def _update_status(self, reached, coll, trunc):
        self._status = np.zeros(self.n, dtype=np.int8)
        self._status[reached] = 1
        self._status[coll] = 2

    def _handle_walls(self):
        """Clip positions to arena and return boolean mask of who hit a wall."""
        half = ARENA_SIZE / 2
        hit_x = np.abs(self.pos[:, 0]) > half - self.radii
        hit_y = np.abs(self.pos[:, 1]) > half - self.radii
        wall_hit = hit_x | hit_y

        if wall_hit.any():
            self.pos[wall_hit, 0] = np.clip(
                self.pos[wall_hit, 0],
                -half + self.radii[wall_hit],
                half - self.radii[wall_hit]
            )
            self.pos[wall_hit, 1] = np.clip(
                self.pos[wall_hit, 1],
                -half + self.radii[wall_hit],
                half - self.radii[wall_hit]
            )
        return wall_hit

    # ------------------------------------------------------------------
    # REWARD FUNCTION  (potential‑based shaping)
    # ------------------------------------------------------------------

    def _compute_reward(self, actions, reached, coll, wall_hit, prev_goal_dist, trunc):
        n = self.n

        # precompute distances & progress
        curr_dist = np.linalg.norm(self.goal - self.pos, axis=1)
        progress = prev_goal_dist - curr_dist

        # nearest‐neighbor surface (drone‐drone)
        rel_all = self.pos[:, None, :] - self.pos[None, :, :]
        dists = np.linalg.norm(rel_all, axis=-1)
        radii_sum = self.radii[None, :] + self.radii[:, None]
        surf_mat = dists - radii_sum
        np.fill_diagonal(surf_mat, np.inf)
        nn_surf = surf_mat.min(axis=1)  # shape (n,)

        rew = np.zeros(n, dtype=np.float32)

        for i in range(n):
            # ── terminal cases ────────────────────────────────────
            if reached[i]:
                rew[i] += REACH_BONUS
                continue
            if coll[i] or wall_hit[i]:
                rew[i] -= COLLISION_PEN
                continue
            if trunc[i]:
                rew[i] -= TRUNC_PEN
                continue

            # 1) progress + tiny costs
            rew[i] += PROG_COEF * progress[i]
            rew[i] -= STEP_COST
            rew[i] -= ACTION_PEN * np.linalg.norm(actions[i])

            # 2) geometry & alignment
            g = self.goal[i] - self.pos[i]
            dist = np.linalg.norm(g) + eps
            ghat = g / dist
            v = self.vel[i]
            speed = np.linalg.norm(v)

            #   a) thrust alignment
            adir = actions[i] / (np.linalg.norm(actions[i]) + eps)
            rew[i] += ALIGN_COEF * np.dot(adir, ghat)
            #   b) velocity alignment
            vdir = v / (speed + eps)
            rew[i] += ALIGN_COEF * np.dot(vdir, ghat)

            #   c) small “speed in clear” bonus
            if dist > BRAKE_R and nn_surf[i] > SAFE_DIST:
                rew[i] += 0.2 * speed

            # 3) final‐mile funnel & braking
            if dist < FUNNEL_R:
                # 3a) well pull
                rew[i] += (1.0 - (dist / FUNNEL_R) ** 2)
                # 3b) braking bonus
                rad_acc = np.dot(actions[i], ghat)
                if dist < BRAKE_R and rad_acc < 0:
                    rew[i] += BRAKE_COEF * (-rad_acc)
                # 3c) stall nudge
                if speed < MIN_PUSH_SPEED:
                    rew[i] -= ZERO_SPEED_PEN
                # 3d) discourage overspeed
                if dist < ALIGN_R and speed > MIN_PUSH_SPEED * 3:
                    rew[i] -= 2.0 * (speed - MIN_PUSH_SPEED * 3)

            # 4) tangential penalty (always)
            vrad = np.dot(v, ghat)
            tang = v - vrad * ghat
            rew[i] -= 0.5 * np.linalg.norm(tang)

            # 5) drone‐drone repulsion (fade inside funnel)
            rep_scale = np.clip(dist / FUNNEL_R, 0.0, 1.0) ** 1.2
            mn = nn_surf[i]
            if mn < SAFE_DIST:
                rew[i] -= rep_scale * REP_COEF * (SAFE_DIST - mn) / SAFE_DIST
            if mn < DANGEROUS_DIST:
                rew[i] -= rep_scale * COLLISION_PEN * (DANGEROUS_DIST - mn) / DANGEROUS_DIST
            # 5b) active dodge away‐velocity
            #    find the argmin j again
            j = np.argmin(surf_mat[i])
            dir_ij = (self.pos[i] - self.pos[j]) / (dists[i, j] + eps)
            toward_j = np.dot(v, dir_ij)
            if toward_j < 0:
                rew[i] += AWAY_DRONE_COEF * (-toward_j)

            # 6) obstacle repulsion & dodge (always, fade in funnel)
            if self.num_obstacles > 0:
                odists = np.linalg.norm(self.obs_pos - self.pos[i], axis=1)
                k = np.argmin(odists)
                surf_o = odists[k] - (self.radii[i] + self.obs_rad[k])

                obs_rep_scale = np.clip(dist / FUNNEL_R, 0.0, 1.0) ** 1.2
                if surf_o < SAFE_DIST:
                    rew[i] -= obs_rep_scale * REP_COEF * (SAFE_DIST - surf_o) / SAFE_DIST
                if surf_o < DANGEROUS_DIST:
                    rew[i] -= obs_rep_scale * COLLISION_PEN * (DANGEROUS_DIST - surf_o) / DANGEROUS_DIST

                # active dodge from obstacle
                dir_io = (self.pos[i] - self.obs_pos[k]) / (odists[k] + eps)
                toward_o = np.dot(v, dir_io)
                if toward_o < 0:
                    rew[i] += AWAY_OBS_COEF * (-toward_o)

        return rew

    # !!! Previous Configurations, works better now
    # def _compute_reward(self, actions, reached, coll, wall_hit, prev_goal_dist, trunc):
    #     SAFE_DIST = 12.0  # start soft repelling at 12 m
    #     DANGEROUS_DIST = 6.0  # hard penalty inside 6 m
    #     FUNNEL_R = 15.0  # final‐mile well radius
    #     BRAKE_R = 12.0
    #     ALIGN_R = 15.0
    #     MAX_SLOW_SPEED = 1.5
    #     MAX_SLOW_SPEED = 1.5  # cap speed inside ALIGN_R
    #     MIN_PUSH_SPEED = 0.5  # if below this in final mile, small penalty
    #
    #     # core coefficients
    #     PROG_COEF = 5.0  # stronger progress reward
    #     STEP_COST = 0.01  # small per‐step penalty
    #     ACTION_PEN = 0.005  # lightly penalize large actions
    #     ALIGN_COEF = 0.5  # keep heading aligned
    #     BRAKE_COEF = 2.0  # mild braking bonus
    #     REP_COEF = 2.0  # light soft‐repulsion
    #     COLLISION_PEN = 15.0  # moderate collision penalty
    #     REACH_BONUS = 50.0  # strong end‐of‐episode bonus
    #     ZERO_SPEED_PEN = 1.0
    #
    #     eps = 1e-8
    #     curr_dist = np.linalg.norm(self.goal - self.pos, axis=1)
    #     progress = prev_goal_dist - curr_dist
    #     rew = np.zeros(self.n, dtype=np.float32)
    #
    #     rel_all = self.pos[:, None, :] - self.pos[None, :, :]  # (n,n,2)
    #     dists = np.linalg.norm(rel_all, axis=-1)  # (n,n)
    #     radii = self.radii[None, :] + self.radii[:, None]  # (n,n)
    #     surf_mat = dists - radii  # (n,n)
    #     np.fill_diagonal(surf_mat, np.inf)
    #     mn_surf = surf_mat.min(axis=1)
    #
    #     for i in range(self.n):
    #         # ── terminal cases ─────────────────────────────────────
    #         if reached[i]:
    #             rew[i] += REACH_BONUS
    #             continue
    #         if coll[i] or wall_hit[i] or trunc[i]:
    #             rew[i] -= COLLISION_PEN
    #             continue
    #
    #         # ── 1) progress & small costs ──────────────────────────
    #         rew[i] += PROG_COEF * progress[i]
    #         rew[i] -= STEP_COST
    #         rew[i] -= ACTION_PEN * np.linalg.norm(actions[i])
    #
    #         # ── 2) goal geometry & alignment ──────────────────────
    #         g = self.goal[i] - self.pos[i]
    #         dist = np.linalg.norm(g) + eps
    #         ghat = g / dist
    #         v = self.vel[i]
    #         speed = np.linalg.norm(v)
    #
    #         # 2a) keep your thrust pointed at the goal
    #         adir = actions[i] / (np.linalg.norm(actions[i]) + eps)
    #         cos_a = np.dot(adir, ghat)
    #         rew[i] += ALIGN_COEF * cos_a
    #
    #         # 2b) keep your velocity pointed at the goal
    #         vdir = v / (speed + eps)
    #         cos_v = np.dot(vdir, ghat)
    #         rew[i] += ALIGN_COEF * cos_v
    #         # ── 2.5) speed boost in the clear ───────────────────
    #         # if we’re safely away from any drone (and obstacle) and
    #         # not in final‐mile braking, reward going fast
    #         mn = mn_surf[i]
    #         if dist > BRAKE_R and mn > SAFE_DIST:
    #             # small bonus proportional to forward speed
    #             rew[i] += 0.2 * speed
    #
    #         # ── 3) final‐mile “well” & braking ─────────────────────
    #         if dist < FUNNEL_R:
    #             # gently pull into the well
    #             rew[i] += 1.0 * (1.0 - (dist / FUNNEL_R) ** 2)
    #             # bonus for braking toward the goal
    #             rad_acc = np.dot(actions[i], ghat)
    #             if rad_acc < 0 and dist < BRAKE_R:
    #                 rew[i] += BRAKE_COEF * (-rad_acc)
    #             # discourage going too fast if very close
    #             if dist < ALIGN_R and speed > MAX_SLOW_SPEED:
    #                 rew[i] -= 2.0 * (speed - MAX_SLOW_SPEED)
    #
    #         # ── 4) simple tangential penalty ───────────────────────
    #         vrad = np.dot(v, ghat)
    #         tang = v - vrad * ghat
    #         rew[i] -= 0.5 * np.linalg.norm(tang)
    #
    #         # ── 5) pairwise drone soft‐repulsion ──────────────────
    #         rel = self.pos - self.pos[i]  # (n,2)
    #         d0 = np.linalg.norm(rel, axis=1) + eps
    #         d0[i] = np.inf
    #         surf = d0 - (self.radii[i] + self.radii)
    #         mn = surf.min()
    #         if mn < SAFE_DIST:
    #             rew[i] -= REP_COEF * (SAFE_DIST - mn) / SAFE_DIST
    #         if mn < DANGEROUS_DIST:
    #             rew[i] -= COLLISION_PEN * (DANGEROUS_DIST - mn) / DANGEROUS_DIST
    #
    #         # ── 6) obstacle repulsion (only outside final‐mile) ──
    #         if self.num_obstacles > 0:
    #             od = np.linalg.norm(self.obs_pos - self.pos[i], axis=1)
    #             k = np.argmin(od)
    #             surf_o = od[k] - (self.radii[i] + self.obs_rad[k])
    #             if surf_o < SAFE_DIST and curr_dist[i] > FUNNEL_R:
    #                 rew[i] -= REP_COEF * (SAFE_DIST - surf_o) / SAFE_DIST
    #             if surf_o < DANGEROUS_DIST:
    #                 rew[i] -= COLLISION_PEN * (DANGEROUS_DIST - surf_o) / DANGEROUS_DIST
    #
    #     return rew

    # ---------- Rendering ------------------------------------------------
    def render(self, mode: str | None = None):
        mode = mode or getattr(self, "render_mode", "human")

        # ------------------------------ constants
        scale, PAD = 10, 50
        EXTRA_H = 100  # text strip height
        arena_px = int(ARENA_SIZE * scale) + 2 * PAD  # square arena + margins
        win_w, win_h = arena_px, arena_px + EXTRA_H

        if not hasattr(self, "_pg_init"):
            pygame.init()
            pygame.font.init()
            self._font = pygame.font.Font(None, 30)  # built‑in font

            if mode == "human":
                self._screen = pygame.display.set_mode((win_w, win_h))
                pygame.display.set_caption("Multi‑Drone RL viewer")
            else:  # off‑screen only
                self._screen = None

            self._clock = pygame.time.Clock()
            self._pg_init = True
            self._win_w, self._win_h = win_w, win_h
            self._arena_px = arena_px

        # ------------------------------ draw to surface
        surf = pygame.Surface((self._win_w, self._win_h))
        surf.fill((30, 30, 30))
        surf.blit(self._font.render(str(self.steps), True, (255, 255, 255)), (3, 3))
        surf.blit(self._font.render(f"Seed = {str(self.seed)}", True, (255, 255, 255)), (53, 3))
        # arena border
        pygame.draw.rect(
            surf, (200, 200, 200),
            (PAD, PAD, self._arena_px - 2 * PAD, self._arena_px - 2 * PAD), width=2
        )

        # obstacles
        for c, r in zip(self.obs_pos, self.obs_rad):
            ox, oy = (c + ARENA_SIZE / 2) * scale + PAD
            pygame.draw.circle(surf, (100, 100, 100),
                               (int(ox), int(oy)), int(r * scale))

        # drones & goals
        for i, (p, v) in enumerate(zip(self.pos, self.vel)):
            col = self.palette[i % len(self.palette)]

            # goal cross
            gx, gy = (self.goal[i] + ARENA_SIZE / 2) * scale + PAD
            gx, gy = int(gx), int(gy)
            pygame.draw.line(surf, col, (gx - 6, gy - 6), (gx + 6, gy + 6), 2)
            pygame.draw.line(surf, col, (gx - 6, gy + 6), (gx + 6, gy - 6), 2)

            # drone disc
            dx, dy = (p + ARENA_SIZE / 2) * scale + PAD
            dx, dy = int(dx), int(dy)
            pygame.draw.circle(surf, col, (dx, dy), int(self.radii[i] * scale))

            # velocity arrow
            vx, vy = v * scale * 0.5
            pygame.draw.line(surf, (255, 255, 255),
                             (dx, dy), (int(dx + vx), int(dy + vy)), 2)
            reward_text = f"{self.last_reward[i]:+.1f}"  # e.g. "±12.3"
            txt_surf = self._font.render(reward_text, True, (255, 255, 255))
            # place it 10px above the drone center
            surf.blit(txt_surf, (dx - txt_surf.get_width() / 2, dy - 20))

            # ------------------------------ LIDAR visualization
            if self.lidar:
                lidar_dists = self._get_lidar(p, i) * 20.0
                n_beams = len(lidar_dists)
                angles = np.linspace(0, 2 * np.pi, n_beams, endpoint=False)

                # Pre-create one surface for all beams
                lidar_surf = pygame.Surface((self._win_w, self._win_h), pygame.SRCALPHA)
                lidar_surf.set_alpha(170)

                for angle, dist in zip(angles, lidar_dists):
                    end_x = dx + np.cos(angle) * dist * scale
                    end_y = dy + np.sin(angle) * dist * scale

                    if dist < 20.0:
                        color = (255, 0, 0, 180) if dist > 2.0 else (255, 100, 100, 180)
                        pygame.draw.circle(surf, (255, 255, 0), (int(end_x), int(end_y)), 3)
                    else:
                        color = (0, 100, 255, 100)

                    pygame.draw.line(lidar_surf, color, (dx, dy), (end_x, end_y), 1)

                surf.blit(lidar_surf, (0, 0))
        # ------------------------------ status panel (multi-column) ------------
        ROWS_PER_COL = 4
        LINE_H = 30
        COL_W = 350

        status_colors = [
            (255, 255, 255),  # live
            (0, 255, 0),  # reached
            (255, 0, 0),  # collision
            (255, 165, 0)  # time-limit
        ]
        y0 = self._arena_px - 40
        x0 = PAD

        for i in range(self.n):
            # grid position
            row, col = i % ROWS_PER_COL, i // ROWS_PER_COL
            s = self._status[i] if hasattr(self, "_status") else 0

            # base drone color & status text
            drone_col = self.palette[i % len(self.palette)]
            status_txt = ["In progress", "Reached goal", "Collision", "Time limit"][s]
            status_col = status_colors[s]

            # compute speed
            speed = np.linalg.norm(self.vel[i])
            # append to status
            status_txt = f"{speed:.2f} m/s | {status_txt}"

            # render two chunks: "Drone #i:" in drone_col, then status+speed
            head = self._font.render(f"Drone #{i}: ", True, drone_col)
            stat = self._font.render(status_txt, True, status_col)

            x = x0 + col * COL_W
            y = y0 + row * LINE_H
            surf.blit(head, (x, y))
            surf.blit(stat, (x + head.get_width(), y))

        # ------------------------------ present or return
        if mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.close()
                    raise KeyboardInterrupt("Viewer closed by user")

            self._screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.event.pump()
            self._clock.tick(30)
            return None

        elif mode == "rgb_array":
            return pygame.surfarray.array3d(surf).swapaxes(0, 1).copy()

        else:
            raise NotImplementedError(f"render mode {mode} not supported")

    def calc_braking_distance(self, speed=5.0):
        dist = 0.0
        while speed > 0:
            dist += speed * DT
            speed -= 2.0 * DT
        return dist

    def close(self):
        if hasattr(self, "_pg_init"):
            pygame.display.quit()
            pygame.quit()
            del self._pg_init


if __name__ == '__main__':
    env = DroneWorldEnv(num_drones=2, num_obstacles=4, max_steps=300, render_mode="human", lidar=False,
                        spawn_area=40.0)
    # print(env.calc_braking_distance(5.0))
    obs, _ = env.reset()
    for k in range(20000):
        # one random action per drone
        env.reset()
        time.sleep(0.5)
        for i in range(env.n):
            pos_i = env.pos[i]
            goal_i = env.pos[i]
            print(f"Dist for {i}: {np.linalg.norm(goal_i - pos_i)}")
        # if k == 400:
        #     env.spawn_area = ARENA_SIZE
        actions = np.stack([env.action_space.sample() for _ in range(env.n)])
        obs, rew, done, trunc, inf = env.step(actions)
        # print(rew)
        # print(inf)
        # print(obs)
        # pos = env.pos[0]
        # print(env.get_lidar(pos,0))
        env.render()
        # time.sleep(1)
        # if k == 100:
        #     frame = env.render()  # returns (H,W,3) uint8
        #     imwrite("snapshot.png", frame)
        # print(env.render())       # comment out during training
        if done.all() or trunc.any():
            break

    env.close()
