# def _compute_reward(self, actions, reached, coll, wall_hit, prev_goal_dist, trunc):
    #     SAFE_DIST = 8.0
    #     eps = 1e-6
    #
    #     curr_dist = np.linalg.norm(self.goal - self.pos, axis=1)
    #     progress = prev_goal_dist - curr_dist
    #     rew = np.zeros(self.n, dtype=np.float32)
    #
    #     for i in range(self.n):
    #         if reached[i]:
    #             rew[i] += 30.0  # bigger terminal bonus
    #             continue
    #         if coll[i] or wall_hit[i]:
    #             rew[i] -= 25.0
    #             continue
    #         if trunc[i]:
    #             rew[i] -= 10.0
    #             continue
    #
    #         # ————— basic shaping —————
    #         rew[i] += 4.0 * progress[i]  # stronger pull downhill
    #         rew[i] -= 0.02  # lighter step cost
    #         rew[i] -= 0.01 * np.linalg.norm(actions[i])
    #
    #         # ————— goal‐vector alignment & radial boost —————
    #         g = self.goal[i] - self.pos[i]
    #         dist = np.linalg.norm(g) + eps
    #         ghat = g / dist
    #         rad_v = np.dot(self.vel[i], ghat)  # radial velocity
    #
    #         rew[i] += 0.5 * rad_v  # reward any forward motion
    #         # tangential‐velocity penalty (stronger near goal)
    #         tang = self.vel[i] - rad_v * ghat
    #         tang_w = np.linalg.norm(tang)
    #         rew[i] -= (0.2 + 0.8 * (curr_dist[i] < 5.0)) * tang_w
    #
    #         # ————— strong in-zone bonus —————
    #         if curr_dist[i] < 2.0:
    #             # a smooth funnel from 2m → 0m
    #             rew[i] += 2.0 * (2.0 - curr_dist[i])
    #
    #         # ————— drone‐drone repulsion —————
    #         rel = self.pos - self.pos[i]
    #         d0 = np.linalg.norm(rel, axis=1) + eps
    #         d0[i] = np.inf
    #         surf = d0 - (self.radii[i] + self.radii)
    #         min_s = surf.min()
    #         if min_s < SAFE_DIST:
    #             rew[i] -= 4.0 * (SAFE_DIST - min_s)
    #             rew[i] -= 0.2 / ((min_s + eps) ** 2)
    #
    #         # ————— obstacle repulsion (thresholded) —————
    #         if self.num_obstacles > 0:
    #             od = np.linalg.norm(self.obs_pos - self.pos[i], axis=1)
    #             k = np.argmin(od)
    #             surf_o = od[k] - (self.radii[i] + self.obs_rad[k])
    #             if surf_o < SAFE_DIST:
    #                 rew[i] -= 1.0 * (SAFE_DIST - surf_o)
    #                 rew[i] -= 0.2 / ((surf_o + eps) ** 2)
    #
    #     return rew

    # def _compute_reward(self, actions, reached, coll, wall_hit, prev_goal_dist, trunc):
    #     # Tunables
    #     SAFE_DIST = 12.0
    #     DANGEROUS_DIST = 6.0
    #     FUNNEL_R = 12.0
    #     BRAKE_R = 9.0
    #     ALIGN_R = 15.0
    #     MAX_SLOW_SPEED = 1.5
    #     MIN_PUSH_SPEED = 0.5
    #     NEAR_MISS_MARGIN = 2.0
    #
    #     # Coefficients
    #     PROG_COEF = 5.0
    #     STEP_COST = 0.01
    #     ACTION_PEN = 0.005
    #     ALIGN_COEF = 0.5
    #     BRAKE_COEF = 3.0
    #     REP_COEF = 3.5
    #     COLLISION_PEN = 80.0
    #     REACH_BONUS = 150.0
    #     ZERO_SPEED_PEN = 3.0
    #     TRUNC_PEN = 300
    #
    #     eps = 1e-8
    #     n = self.n
    #
    #     curr_dist = np.linalg.norm(self.goal - self.pos, axis=1)
    #     progress = prev_goal_dist - curr_dist
    #
    #     rel_all = self.pos[:, None, :] - self.pos[None, :, :]
    #     dists = np.linalg.norm(rel_all, axis=-1)
    #     radii_sum = self.radii[None, :] + self.radii[:, None]
    #     surf_mat = dists - radii_sum
    #     np.fill_diagonal(surf_mat, np.inf)
    #     nn_surf = surf_mat.min(axis=1)
    #
    #     rew = np.zeros(n, dtype=np.float32)
    #
    #     for i in range(n):
    #         if reached[i]:
    #             rew[i] += REACH_BONUS
    #             continue
    #         if coll[i] or wall_hit[i] :
    #             rew[i] -= COLLISION_PEN
    #             continue
    #         if trunc[i]:
    #             rew[i] -= TRUNC_PEN
    #             continue
    #
    #         # Progress + base costs
    #         rew[i] += PROG_COEF * progress[i]
    #         rew[i] -= STEP_COST
    #         rew[i] -= ACTION_PEN * np.linalg.norm(actions[i])
    #
    #         # Goal vector
    #         g = self.goal[i] - self.pos[i]
    #         dist = np.linalg.norm(g) + eps
    #         ghat = g / dist
    #         v = self.vel[i]
    #         speed = np.linalg.norm(v)
    #
    #         # Alignment: thrust + velocity
    #         adir = actions[i] / (np.linalg.norm(actions[i]) + eps)
    #         vdir = v / (speed + eps)
    #         rew[i] += ALIGN_COEF * np.dot(adir, ghat)
    #         rew[i] += ALIGN_COEF * np.dot(vdir, ghat)
    #
    #         # Speed-in-the-clear bonus
    #         if dist > BRAKE_R and nn_surf[i] > SAFE_DIST:
    #             rew[i] += 0.2 * speed
    #
    #         # Final-mile funnel logic
    #         if dist < FUNNEL_R:
    #             rew[i] += (1.0 - (dist / FUNNEL_R) ** 2)
    #             rad_acc = np.dot(actions[i], ghat)
    #             if dist < BRAKE_R and rad_acc < 0:
    #                 rew[i] += BRAKE_COEF * (-rad_acc)
    #             if speed < MIN_PUSH_SPEED:
    #                 rew[i] -= ZERO_SPEED_PEN
    #             if dist < ALIGN_R and speed > MAX_SLOW_SPEED:
    #                 rew[i] -= 2.0 * (speed - MAX_SLOW_SPEED)
    #
    #         # Tangential motion penalty (stronger in funnel)
    #         vrad = np.dot(v, ghat)
    #         tang = v - vrad * ghat
    #         tang_penalty = 1.5 if dist < FUNNEL_R else 0.5
    #         rew[i] -= tang_penalty * np.linalg.norm(tang)
    #
    #         # Scale conflict penalty based on distance to nearest drone
    #         if self.n > 1:
    #             drone_dists = np.linalg.norm(self.pos - self.pos[i], axis=1)
    #             drone_dists[i] = np.inf
    #             j = np.argmin(drone_dists)
    #             vec_to_drone = self.pos[j] - self.pos[i]
    #             dist_to_drone = np.linalg.norm(vec_to_drone)
    #             norm_vec = vec_to_drone / (dist_to_drone + eps)
    #             heading = v / (speed + eps)
    #             danger_alignment = np.dot(heading, norm_vec)
    #             if self.n > 1:
    #                 rel_vel = self.vel[j] - self.vel[i]
    #                 closing = np.dot(rel_vel, norm_vec)
    #                 if closing > 0:
    #                     rew[i] -= 0.1 * closing  # very mild, non-terminal
    #
    #             # More sensitive to angular collision risk
    #             if dist_to_drone < SAFE_DIST and danger_alignment > 0.5:
    #                 weight = (danger_alignment - 0.5) * 2.0  # scale from 0 to 1
    #                 factor = np.clip((SAFE_DIST - dist_to_drone) / SAFE_DIST, 0.0, 1.0)
    #                 rew[i] -= 0.8 * weight * factor  # more continuous effect
    #
    #
    #         # Drone repulsion (always full)
    #         mn = nn_surf[i]
    #         if mn < SAFE_DIST:
    #             rew[i] -= REP_COEF * (SAFE_DIST - mn) / SAFE_DIST
    #         if mn < DANGEROUS_DIST:
    #             rew[i] -= COLLISION_PEN * (DANGEROUS_DIST - mn) / DANGEROUS_DIST
    #         if DANGEROUS_DIST <= mn < DANGEROUS_DIST + NEAR_MISS_MARGIN:
    #             rew[i] -= 0.5  # Near-miss penalty
    #
    #         # Obstacle repulsion (stronger)
    #         if self.num_obstacles > 0:
    #             odists = np.linalg.norm(self.obs_pos - self.pos[i], axis=1)
    #             k = np.argmin(odists)
    #             surf_o = odists[k] - (self.radii[i] + self.obs_rad[k])
    #             if surf_o < SAFE_DIST:
    #                 rew[i] -= 2.0 * REP_COEF * (SAFE_DIST - surf_o) / SAFE_DIST
    #             if surf_o < DANGEROUS_DIST:
    #                 rew[i] -= 2.0 * COLLISION_PEN * (DANGEROUS_DIST - surf_o) / DANGEROUS_DIST
    #
    #     return rew

    # def _compute_reward(self, actions, reached, coll, wall_hit, prev_goal_dist, trunc):
    #     # ───────────── tunables ────────────────────────────────────
    #     SAFE_DIST = 12.0  # start soft repelling at 12 m
    #     DANGEROUS_DIST = 6.0  # hard penalty inside 6 m
    #     FUNNEL_R = 15.0  # final-mile “funnel” radius
    #     BRAKE_R = 9.0  # start braking bonus here
    #     ALIGN_R = 15.0  # heading alignment radius
    #     MAX_SLOW_SPEED = 1.5  # cap speed inside ALIGN_R
    #     MIN_PUSH_SPEED = 0.5  # if slower than this in funnel, add nudge
    #
    #     # core coefficients
    #     PROG_COEF = 5.0
    #     STEP_COST = 0.01
    #     ACTION_PEN = 0.005
    #     ALIGN_COEF = 0.5
    #     BRAKE_COEF = 2.0
    #     REP_COEF = 2.0
    #     COLLISION_PEN = 15.0
    #     REACH_BONUS = 150.0
    #     ZERO_SPEED_PEN = 1.0
    #
    #     eps = 1e-8
    #     n = self.n
    #
    #     # 0) distances & progress
    #     curr_dist = np.linalg.norm(self.goal - self.pos, axis=1)
    #     progress = prev_goal_dist - curr_dist
    #
    #     # 0.5) precompute nearest‐neighbor surface distances
    #     rel_all = self.pos[:, None, :] - self.pos[None, :, :]  # (n,n,2)
    #     dists = np.linalg.norm(rel_all, axis=-1)  # (n,n)
    #     radii_sum = self.radii[None, :] + self.radii[:, None]  # (n,n)
    #     surf_mat = dists - radii_sum
    #     np.fill_diagonal(surf_mat, np.inf)
    #     nn_surf = surf_mat.min(axis=1)  # shape (n,)
    #
    #     rew = np.zeros(n, dtype=np.float32)
    #
    #     for i in range(n):
    #         # ── terminal cases ─────────────────────────────────────
    #         if reached[i]:
    #             rew[i] += REACH_BONUS
    #             continue
    #         if coll[i] or wall_hit[i] or trunc[i]:
    #             rew[i] -= COLLISION_PEN
    #             continue
    #
    #         # 1) progress + minimal costs
    #         rew[i] += PROG_COEF * progress[i]
    #         rew[i] -= STEP_COST
    #         rew[i] -= ACTION_PEN * np.linalg.norm(actions[i])
    #
    #         # 2) geometry & alignment
    #         g = self.goal[i] - self.pos[i]
    #         dist = np.linalg.norm(g) + eps
    #         ghat = g / dist
    #         v = self.vel[i]
    #         speed = np.linalg.norm(v)
    #
    #         #   a) thrust alignment
    #         adir = actions[i] / (np.linalg.norm(actions[i]) + eps)
    #         rew[i] += ALIGN_COEF * np.dot(adir, ghat)
    #
    #         #   b) velocity alignment
    #         vdir = v / (speed + eps)
    #         rew[i] += ALIGN_COEF * np.dot(vdir, ghat)
    #
    #         #   c) speed-in-the-clear bonus
    #         if dist > BRAKE_R and nn_surf[i] > SAFE_DIST:
    #             rew[i] += 0.2 * speed
    #
    #         # 3) final-mile funnel & braking
    #         if dist < FUNNEL_R:
    #             # gentle “well” pull
    #             rew[i] += (1.0 - (dist / FUNNEL_R) ** 2)
    #             # braking bonus
    #             rad_acc = np.dot(actions[i], ghat)
    #             if dist < BRAKE_R and rad_acc < 0:
    #                 rew[i] += BRAKE_COEF * (-rad_acc)
    #             # nudge if you stall in the funnel
    #             if speed < MIN_PUSH_SPEED:
    #                 rew[i] -= ZERO_SPEED_PEN
    #             # discourage overspeed deep in funnel
    #             if dist < ALIGN_R and speed > MAX_SLOW_SPEED:
    #                 rew[i] -= 2.0 * (speed - MAX_SLOW_SPEED)
    #
    #         # 4) simple tangential penalty (always)
    #         vrad = np.dot(v, ghat)
    #         tang = v - vrad * ghat
    #         rew[i] -= 0.5 * np.linalg.norm(tang)
    #
    #         # 5) repulsion — **scaled down** inside the funnel
    #         rep_scale = np.clip(dist / FUNNEL_R, 0.0, 1.0)
    #         mn = nn_surf[i]
    #         if mn < SAFE_DIST:
    #             rew[i] -= rep_scale * REP_COEF * (SAFE_DIST - mn) / SAFE_DIST
    #         if mn < DANGEROUS_DIST:
    #             rew[i] -= rep_scale * COLLISION_PEN * (DANGEROUS_DIST - mn) / DANGEROUS_DIST
    #
    #         # 6) obstacle repulsion only **outside** funnel
    #         # if self.num_obstacles > 0 and dist > FUNNEL_R:
    #         #     odists = np.linalg.norm(self.obs_pos - self.pos[i], axis=1)
    #         #     surf_o = odists.min() - (self.radii[i] + self.obs_rad[odists.argmin()])
    #         #     if surf_o < SAFE_DIST:
    #         #         rew[i] -= REP_COEF * (SAFE_DIST - surf_o) / SAFE_DIST
    #         #     if surf_o < DANGEROUS_DIST:
    #         #         rew[i] -= COLLISION_PEN * (DANGEROUS_DIST - surf_o) / DANGEROUS_DIST
    #
    #         if self.num_obstacles > 0:
    #             odists = np.linalg.norm(self.obs_pos - self.pos[i], axis=1)
    #             k = np.argmin(odists)
    #             surf_o = odists[k] - (self.radii[i] + self.obs_rad[k])
    #
    #             # scale from 1.0 when far away, → 0.0 at dist==0
    #             rep_scale = np.clip(dist / FUNNEL_R, 0.0, 1.0) ** 2
    #
    #             if surf_o < SAFE_DIST:
    #                 rew[i] -= rep_scale * REP_COEF * (SAFE_DIST - surf_o) / SAFE_DIST
    #             if surf_o < DANGEROUS_DIST:
    #                 rew[i] -= rep_scale * COLLISION_PEN * (DANGEROUS_DIST - surf_o) / DANGEROUS_DIST
    #
    #     return rew

    # def _compute_reward(self, actions, reached, coll, wall_hit, prev_goal_dist, trunc):
    #     # ───────────── tunables ────────────────────────────────────
    #     SAFE_DIST = 12.0  # start soft repelling at 12 m
    #     DANGEROUS_DIST = 6.0  # hard penalty inside 6 m
    #     FUNNEL_R = 15.0  # final-mile “funnel” radius
    #     BRAKE_R = 12.0  # start braking bonus here
    #     ALIGN_R = 15.0  # heading bonus radius
    #     MIN_PUSH_SPEED = 0.5  # if you go slower than this in the funnel, push you
    #
    #     # core coefficients
    #     PROG_COEF = 5.0
    #     STEP_COST = 0.02
    #     ACTION_PEN = 0.01
    #     ALIGN_COEF = 0.5
    #     BRAKE_COEF = 4.0
    #     REP_COEF = 2.0
    #     DANG_PEN_COEF = 5.0
    #     COLLISION_PEN = 150.0
    #     REACH_BONUS = 50.0
    #     ZERO_SPEED_PEN = 0.1  # per-step stalling penalty inside funnel
    #
    #     eps = 1e-8
    #     n = self.n
    #
    #     # distances
    #     curr_dist = np.linalg.norm(self.goal - self.pos, axis=1)
    #     progress = prev_goal_dist - curr_dist
    #
    #     # pre-compute pairwise surf distances once
    #     rel_all = self.pos[:, None, :] - self.pos[None, :, :]  # (n,n,2)
    #     dists = np.linalg.norm(rel_all, axis=-1)  # (n,n)
    #     radii_sum = self.radii[None, :] + self.radii[:, None]  # (n,n)
    #     surf_mat = dists - radii_sum  # (n,n)
    #     np.fill_diagonal(surf_mat, np.inf)
    #     mn_surf = surf_mat.min(axis=1)  # min surface to any other drone
    #
    #     rew = np.zeros(n, dtype=np.float32)
    #
    #     for i in range(n):
    #         if reached[i]:
    #             rew[i] += REACH_BONUS
    #             continue
    #         if coll[i] or wall_hit[i] or trunc[i]:
    #             rew[i] -= COLLISION_PEN
    #             continue
    #
    #         # 1) progress + small costs
    #         rew[i] += PROG_COEF * progress[i]
    #         rew[i] -= STEP_COST
    #         rew[i] -= ACTION_PEN * np.linalg.norm(actions[i])
    #
    #         # 2) geometry & alignment
    #         g = self.goal[i] - self.pos[i]
    #         dist = np.linalg.norm(g) + eps
    #         ghat = g / dist
    #         v = self.vel[i]
    #         speed = np.linalg.norm(v)
    #
    #         adir = actions[i] / (np.linalg.norm(actions[i]) + eps)
    #         cos_a = np.dot(adir, ghat)
    #         rew[i] += ALIGN_COEF * cos_a
    #
    #         vdir = v / (speed + eps)
    #         cos_v = np.dot(vdir, ghat)
    #         rew[i] += ALIGN_COEF * cos_v
    #
    #         # 2.5) small speed-boost in the clear
    #         if dist > BRAKE_R and mn_surf[i] > SAFE_DIST:
    #             rew[i] += 0.1 * speed
    #
    #         # 3) final-mile funnel & braking
    #         if dist < FUNNEL_R:
    #             # gentle “well” pull
    #             rew[i] += 1.0 * (1.0 - (dist / FUNNEL_R) ** 2)
    #             # braking bonus
    #             rad_acc = np.dot(actions[i], ghat)
    #             if rad_acc < 0 and dist < BRAKE_R:
    #                 rew[i] += BRAKE_COEF * (-rad_acc)
    #             # zero-speed nudge
    #             if speed < MIN_PUSH_SPEED:
    #                 rew[i] -= ZERO_SPEED_PEN
    #             # discourage too-fast near the goal
    #             if dist < ALIGN_R and speed > MIN_PUSH_SPEED * 3:
    #                 rew[i] -= 2.0 * (speed - MIN_PUSH_SPEED * 3)
    #
    #         # 4) simple tangential penalty
    #         vrad = np.dot(v, ghat)
    #         tang = v - vrad * ghat
    #         rew[i] -= 0.5 * np.linalg.norm(tang)
    #
    #         # repulsion scaling: 1.0 far from goal → 0.0 at goal
    #         rep_scale = np.clip(dist / FUNNEL_R, 0.0, 1.0)
    #
    #         # 5) drone-drone soft repulsion
    #         mn = mn_surf[i]
    #         if mn < SAFE_DIST:
    #             rew[i] -= rep_scale * REP_COEF * (SAFE_DIST - mn) / SAFE_DIST
    #         if mn < DANGEROUS_DIST:
    #             rew[i] -= rep_scale * DANG_PEN_COEF * (DANGEROUS_DIST - mn) / DANGEROUS_DIST
    #
    #         # 6) obstacle repulsion (only outside funnel)
    #         if self.num_obstacles > 0:
    #             odists = np.linalg.norm(self.obs_pos - self.pos[i], axis=1)
    #             k = np.argmin(odists)
    #             surf_o = odists[k] - (self.radii[i] + self.obs_rad[k])
    #             if surf_o < SAFE_DIST:
    #                 rew[i] -= rep_scale * REP_COEF * (SAFE_DIST - surf_o) / SAFE_DIST
    #             if surf_o < DANGEROUS_DIST:
    #                 rew[i] -= rep_scale * DANG_PEN_COEF * (DANGEROUS_DIST - surf_o) / DANGEROUS_DIST
    #
    #     return rew

    # def _compute_reward(self, actions, reached, coll, wall_hit,
    #                     prev_goal_dist, trunc):
    #     # ───── tunables ───────────────────────────────────────────────
    #     SAFE_DIST = 10.0  # repulsion starts here (m)
    #     DANGEROUS_DIST = 4.0  # hard penalty inside here (m)
    #     FUNNEL_R = 15.0  # final‐mile “well” radius (m)
    #     BRAKE_R = 12.0  # start rewarding braking at (m)
    #     ALIGN_R = 15.0  # in this radius, require tight alignment
    #     MAX_SLOW_SPEED = 1.5  # cap speed inside ALIGN_R
    #
    #     # core shaping coeffs
    #     PROG_COEF = 4.0
    #     STEP_COST = 0.02
    #     ACTION_PEN = 0.01
    #     ALIGN_ACT_LIN = 0.3
    #     ALIGN_ACT_SQR = 0.7
    #     ANG_PEN_COEF = 1.0
    #     BRAKE_COEF = 4.0
    #     DETOUR_COEF = 1.0
    #
    #     # repulsion / separation
    #     DRONE_REP_K = 5.0  # inverse‐square strength
    #     DRONE_AWAY_K = 2.0  # reward for moving away
    #     OBS_REP_K = 3.0
    #     OBS_DANG_PEN_K = 5.0
    #     DANG_PEN_COEF = 5.0
    #
    #     eps = 1e-8
    #
    #     curr_dist = np.linalg.norm(self.goal - self.pos, axis=1)
    #     progress = prev_goal_dist - curr_dist
    #     rew = np.zeros(self.n, dtype=np.float32)
    #
    #     for i in range(self.n):
    #         # ── terminal cases ────────────────────────────────────
    #         if reached[i]:
    #             rew[i] += 30.0
    #             continue
    #         if coll[i] or wall_hit[i] or trunc[i]:
    #             rew[i] -= 50.0
    #             continue
    #
    #         # ── 1) basic shaping ──────────────────────────────────
    #         rew[i] += PROG_COEF * progress[i]
    #         rew[i] -= STEP_COST
    #         rew[i] -= ACTION_PEN * np.linalg.norm(actions[i])
    #
    #         # ── 2) goal geometry & alignment ─────────────────────
    #         g = self.goal[i] - self.pos[i]
    #         dist = np.linalg.norm(g) + eps
    #         ghat = g / dist
    #         v = self.vel[i]
    #         speed = np.linalg.norm(v)
    #
    #         # 2a) action alignment
    #         adir = actions[i] / (np.linalg.norm(actions[i]) + eps)
    #         cos_a = np.dot(adir, ghat)
    #         rew[i] += ALIGN_ACT_LIN * cos_a + ALIGN_ACT_SQR * (cos_a ** 2)
    #
    #         # 2b) velocity alignment + angular error
    #         vdir = v / (speed + eps)
    #         cos_v = np.dot(vdir, ghat)
    #         rew[i] += 0.5 * cos_v
    #         rew[i] -= ANG_PEN_COEF * np.arccos(np.clip(cos_v, -1, 1))
    #
    #         # ── 3) final‐mile super‐alignment & cross‐track ──────
    #         if dist < ALIGN_R:
    #             # heavy heading bonus
    #             rew[i] += 5.0 * (cos_v ** 4)
    #             # cross‐track
    #             along = self.pos[i] + ghat * dist
    #             cross = np.linalg.norm(self.pos[i] - along)
    #             rew[i] -= 2.0 * np.clip(cross / ALIGN_R, 0, 1)
    #             # speed cap
    #             if speed > MAX_SLOW_SPEED:
    #                 rew[i] -= 5.0 * (speed - MAX_SLOW_SPEED)
    #
    #         # ── 4) braking bonus & detour ─────────────────────────
    #         # braking
    #         rad_acc = np.dot(actions[i], ghat)
    #         if dist < BRAKE_R and rad_acc < 0:
    #             rew[i] += BRAKE_COEF * (-rad_acc)
    #         # detour
    #         detour = max(speed * DT - progress[i], 0.0)
    #         rew[i] -= DETOUR_COEF * detour
    #
    #         # ── 5) final‐mile “well” & overshoot ──────────────────
    #         vrad = np.dot(v, ghat)
    #         if dist < FUNNEL_R:
    #             # well pull
    #             rew[i] += 2.0 * (1 - (dist / FUNNEL_R) ** 2)
    #             # outward penalty
    #             if vrad < 0:
    #                 rew[i] -= 2.0 * (-vrad)
    #
    #         # ── 6) tangential penalty ─────────────────────────────
    #         tang = v - vrad * ghat
    #         tang_w = np.linalg.norm(tang)
    #         beta = np.clip(1 - dist / FUNNEL_R, 0, 1)
    #         pen_coef = 0.2 + 1.5 * beta
    #         rew[i] -= pen_coef * tang_w
    #
    #         # ── 7) drone–drone repulsion & separation ────────────
    #         rel = self.pos - self.pos[i]  # (n,2)
    #         d0 = np.linalg.norm(rel, axis=1) + eps
    #         d0[i] = np.inf
    #         surf = d0 - (self.radii[i] + self.radii)
    #         j = np.argmin(surf)
    #         mn = surf[j]
    #         # inverse‐square potential (zero at SAFE_DIST)
    #         if mn < SAFE_DIST:
    #             phi = DRONE_REP_K * (1 / (mn + eps) ** 2 - 1 / (SAFE_DIST) ** 2)
    #             rew[i] -= phi
    #         # encourage moving away
    #         dir_ij = rel[j] / (d0[j] + eps)
    #         toward = np.dot(v, dir_ij)
    #         if toward < 0:
    #             rew[i] += DRONE_AWAY_K * (-toward)
    #         # hard penalty if truly too close
    #         if mn < DANGEROUS_DIST:
    #             rew[i] -= DANG_PEN_COEF * (DANGEROUS_DIST - mn) / DANGEROUS_DIST
    #
    #         # ── 8) obstacle repulsion (same shape) ──────────────
    #         if self.num_obstacles > 0:
    #             od = np.linalg.norm(self.obs_pos - self.pos[i], axis=1)
    #             k = np.argmin(od)
    #             surf_o = od[k] - (self.radii[i] + self.obs_rad[k])
    #             if surf_o < SAFE_DIST and curr_dist[i] > FUNNEL_R:
    #                 # smooth
    #                 rew[i] -= OBS_REP_K * (SAFE_DIST - surf_o) / SAFE_DIST
    #             if surf_o < DANGEROUS_DIST:
    #                 rew[i] -= OBS_DANG_PEN_K * (DANGEROUS_DIST - surf_o) / DANGEROUS_DIST
    #
    #     return rew

    # def _compute_reward(self, actions, reached, coll, wall_hit, prev_goal_dist, trunc):
    #     # ───── tunables ───────────────────────────────────────────────
    #     SAFE_DIST = 7.0  # start repelling at 7 m
    #     DANGEROUS_DIST = 5.0
    #     FUNNEL_R = 8.0  # final‐mile well radius
    #     BRAKE_DIST = 8.0  # start rewarding braking at 12 m
    #     BRAKE_COEF = 2.0  # strength of braking bonus
    #     DETOUR_COEF = 3.0
    #
    #     eps = 1e-6
    #
    #     # coefficients
    #     PROG_COEF = 4.0
    #     STEP_COST = 0.02
    #     ACTION_PENALTY = 0.01
    #     ALIGN_ACT_LIN = 0.3
    #     ALIGN_ACT_SQR = 0.7
    #     ANG_PEN_COEF = 0.5
    #     DRONE_REP_COEF = 1.5
    #     DANG_PEN_COEF = 2.0
    #
    #     curr_dist = np.linalg.norm(self.goal - self.pos, axis=1)
    #     progress = prev_goal_dist - curr_dist
    #     rew = np.zeros(self.n, dtype=np.float32)
    #
    #     for i in range(self.n):
    #         # ─ terminal cases ─────────────────────────────────────
    #         if reached[i]:
    #             rew[i] += 30.0
    #             continue
    #         if coll[i] or wall_hit[i]:
    #             rew[i] -= 10.0
    #             continue
    #         if trunc[i]:
    #             rew[i] -= 10.0
    #             continue
    #
    #         # ─ basic shaping ──────────────────────────────────────
    #         rew[i] += PROG_COEF * progress[i]
    #         rew[i] -= STEP_COST
    #         rew[i] -= ACTION_PENALTY * np.linalg.norm(actions[i])
    #
    #         # ─ goal geometry ──────────────────────────────────────
    #         g = self.goal[i] - self.pos[i]
    #         dist = np.linalg.norm(g) + eps
    #         ghat = g / dist
    #
    #         # ─ alignment bonuses ──────────────────────────────────
    #         adir = actions[i] / (np.linalg.norm(actions[i]) + eps)
    #         cos_a = np.dot(adir, ghat)
    #         rew[i] += ALIGN_ACT_LIN * cos_a + ALIGN_ACT_SQR * (cos_a ** 2)
    #
    #         vdir = self.vel[i] / (np.linalg.norm(self.vel[i]) + eps)
    #         cos_v = np.dot(vdir, ghat)
    #         rew[i] += 0.5 * cos_v
    #
    #         angle_err = np.arccos(np.clip(cos_v, -1.0, 1.0))
    #         rew[i] -= ANG_PEN_COEF * angle_err
    #
    #         # ─ tangential penalty ────────────────────────────────
    #         vrad = np.dot(self.vel[i], ghat)
    #         tang = self.vel[i] - vrad * ghat
    #         tang_w = np.linalg.norm(tang)
    #         beta = np.clip(1.0 - (curr_dist[i] / FUNNEL_R), 0.0, 1.0)
    #         pen_coef = 0.2 + 0.8 * beta
    #         rew[i] -= pen_coef * tang_w
    #
    #         # ─ outward penalty ────────────────────────────────────
    #         if curr_dist[i] < FUNNEL_R and vrad < 0:
    #             rew[i] -= DANG_PEN_COEF * (-vrad)
    #
    #         # ─ final‐mile well ─────────────────────────────────────
    #         if curr_dist[i] < FUNNEL_R:
    #             rew[i] += 1.0 * (1.0 - (curr_dist[i] / FUNNEL_R) ** 2)
    #
    #         # ─ braking bonus ───────────────────────────────────────
    #         # reward negative radial acceleration (braking) when within BRAKE_DIST
    #         rad_acc = np.dot(actions[i], ghat)
    #         if curr_dist[i] < BRAKE_DIST and rad_acc < 0:
    #             rew[i] += BRAKE_COEF * (-rad_acc)
    #
    #         # ─ path‐optimality penalty ────────────────────────────
    #         # “path_length” this step = speed * DT
    #         # “direct_progress”   = progress[i]
    #         # detour = extra distance traveled beyond direct progress
    #         path_len = np.linalg.norm(self.vel[i]) * DT
    #         direct_prog = progress[i]
    #         detour = max(path_len - direct_prog, 0.0)
    #         rew[i] -= DETOUR_COEF * detour
    #
    #         # ─ drone‐drone repulsion ──────────────────────────────
    #         rel = self.pos - self.pos[i]
    #         d0 = np.linalg.norm(rel, axis=1) + eps
    #         d0[i] = np.inf
    #         surf0 = d0 - (self.radii[i] + self.radii)
    #         min_s = surf0.min()
    #         if min_s < SAFE_DIST:
    #             rew[i] -= DRONE_REP_COEF * (SAFE_DIST - min_s) / SAFE_DIST
    #             if min_s < DANGEROUS_DIST:
    #                 rew[i] -= DANG_PEN_COEF * (DANGEROUS_DIST - min_s)
    #
    #         # ─ obstacle repulsion ────────────────────────────────
    #         if self.num_obstacles > 0:
    #             od = np.linalg.norm(self.obs_pos - self.pos[i], axis=1)
    #             idx_o = np.argmin(od)
    #             surf_o = od[idx_o] - (self.radii[i] + self.obs_rad[idx_o])
    #             if surf_o < SAFE_DIST:
    #                 rew[i] -= DRONE_REP_COEF * (SAFE_DIST - surf_o) / SAFE_DIST
    #                 if surf_o < DANGEROUS_DIST:
    #                     rew[i] -= DANG_PEN_COEF * (DANGEROUS_DIST - surf_o)
    #
    #     return rew

    # def _compute_reward(self, actions, reached, coll, wall_hit, prev_goal_dist, trunc):
    #     eps = 1e-6
    #     curr_dist = np.linalg.norm(self.goal - self.pos, axis=1)
    #     progress = prev_goal_dist - curr_dist
    #     rew = np.zeros(self.n)
    #
    #     for i in range(self.n):
    #         if reached[i]:
    #             rew[i] += 20
    #             continue
    #         if coll[i] or wall_hit[i]:
    #             rew[i] -= 10
    #             continue
    #         if trunc[i]:
    #             rew[i] -= 10
    #             continue
    #
    #         # basic shaping
    #         rew[i] += 4.0 * progress[i]  # negative when moving away
    #         rew[i] -= 0.05  # step cost
    #         rew[i] -= 0.01 * np.linalg.norm(actions[i])  # control cost
    #         if curr_dist[i] < 3.0:
    #             rew[i] += 1.0 * (3.0 - curr_dist[i])
    #         # goal‐direction bonus
    #         g = self.goal[i] - self.pos[i]
    #         gdir = g / (np.linalg.norm(g) + 1e-8)
    #         adir = actions[i] / (np.linalg.norm(actions[i]) + 1e-8)
    #         rew[i] += 0.1 * np.dot(adir, gdir)
    #
    #         # tangential‐velocity penalty
    #         v = self.vel[i]
    #         vrad = np.dot(v, gdir) * gdir
    #         vtan = v - vrad
    #         rew[i] -= 0.2 * np.linalg.norm(vtan)
    #
    #         # agent‐agent repulsion (inverse square)
    #
    #         # inside your loop, replace the inverse‐square block with:
    #         if self.n > 1:
    #             rel = self.pos - self.pos[i]
    #             d0 = np.linalg.norm(rel, axis=1) + 1e-8
    #             d0[i] = np.inf
    #             surf = d0 - (self.radii[i] + self.radii)
    #             j = np.argmin(surf)
    #             min_s = surf[j]
    #
    #             if min_s < SAFE_DIST:
    #                 # linear penalty:
    #                 rew[i] -= 1.0 * (SAFE_DIST - min_s)
    #                 # plus a small inverse‐square for “really critical” avoidance:
    #                 rew[i] -= 0.2 / (max(min_s, eps) ** 2)
    #
    #         # obstacle repulsion (same idea)
    #         if self.num_obstacles > 0:
    #             od = np.linalg.norm(self.obs_pos - self.pos[i], axis=1)
    #             k = np.argmin(od)
    #             surf_o = od[k] - (self.radii[i] + self.obs_rad[k])
    #             if surf_o < SAFE_DIST:
    #                 # linear penalty
    #                 rew[i] -= 1.0 * (SAFE_DIST - surf_o)
    #                 # inverse‐square kick
    #                 rew[i] -= 0.2 / ((surf_o + eps) ** 2)
    #
    #     return rew

    # def _compute_reward(self, actions, reached, coll, wall_hit, prev_goal_dist, trunc):
    #     curr_dist = np.linalg.norm(self.goal - self.pos, axis=1)
    #
    #     # Calculate direction vectors
    #     r_vec = self.goal - self.pos
    #     dists = np.linalg.norm(r_vec, axis=1, keepdims=True) + 1e-8
    #     r_hat = r_vec / dists
    #
    #     # Radial velocity (movement toward goal)
    #     radial_vel = np.sum(self.vel * r_hat, axis=1)
    #
    #     # Tangential movement penalty (curved paths)
    #     v_radial = radial_vel[:, None] * r_hat
    #     v_tangential = self.vel - v_radial
    #     tang_speed = np.linalg.norm(v_tangential, axis=1)
    #
    #     # Progress rewards
    #     progress_reward = radial_vel * DT * PROGRESS_COEF
    #
    #     # Terminal rewards (per-agent)
    #     terminal = np.zeros(self.n, dtype=np.float32)
    #     terminal[reached] = R_GOAL
    #     terminal[coll] = R_COLL
    #     terminal[wall_hit] = R_WALL
    #     terminal[trunc & ~reached] = R_TIMEOUT
    #
    #     # Precision landing bonuses
    #     precision_bonus = np.zeros(self.n)
    #     very_close = curr_dist < 1.5
    #     precision_bonus[very_close] = 5.0 * (1.5 - curr_dist[very_close])
    #
    #     # Velocity alignment bonus near goal
    #     alignment_bonus = np.zeros(self.n)
    #     near_goal = (curr_dist < 5.0) & (curr_dist > 1.0)
    #     if np.any(near_goal):
    #         v_norm = self.vel / (np.linalg.norm(self.vel, axis=1, keepdims=True) + 1e-8)
    #         cosine = np.sum(v_norm * r_hat, axis=1)
    #         alignment_bonus[near_goal] = ALIGNMENT_BONUS_NEAR_GOAL_COEF * cosine[near_goal]
    #
    #     # Tangential penalty - stronger near goal
    #     tang_penalty = np.where(
    #         curr_dist < 3.0,
    #         TANGENTIAL_PENALTY_COEF * 2.0 * tang_speed,
    #         TANGENTIAL_PENALTY_COEF * tang_speed
    #     )
    #
    #     # Base rewards (per-agent)
    #     reward = (
    #             progress_reward +
    #             radial_vel * RADIAL_VEL_COEF +
    #             alignment_bonus +
    #             precision_bonus +
    #             terminal -
    #             tang_penalty -
    #             ACTION_PENALTY_COEF * np.linalg.norm(actions, axis=1) -
    #             TIME_PENALTY
    #     )
    #
    #     for i in range(self.n):
    #         # get raw center-to-center distance
    #         dists = np.linalg.norm(self.pos - self.pos[i], axis=1)
    #         dists[i] = np.inf
    #         min_dist = dists.min() - self.radii[i]
    #
    #         if min_dist < SAFE_DIST:
    #             # linear penalty: the closer they are, the worse
    #             reward[i] -= AVOID_PENALTY * (SAFE_DIST - min_dist) ** 2
    #             if min_dist < DANGEROUS_DIST:
    #                 reward[i] -= AVOID_PENALTY
    #
    #     # Obstacle avoidance incentives
    #     if self.num_obstacles > 0:
    #         for i in range(self.n):
    #             obs_dists = np.linalg.norm(self.obs_pos - self.pos[i], axis=1)
    #             min_obs_dist = np.min(obs_dists) - self.radii[i]
    #
    #             if min_obs_dist < SAFE_DIST:
    #                 # Progressive penalty as distance decreases
    #                 obs_penalty = AVOID_PENALTY * (SAFE_DIST - min_obs_dist) ** 2
    #                 reward[i] -= obs_penalty
    #
    #                 # Strong penalty for near-collision
    #                 if min_obs_dist < DANGEROUS_DIST:
    #                     reward[i] -= AVOID_PENALTY
    #
    #     # Deceleration bonus for smooth landing
    #     speed = np.linalg.norm(self.vel, axis=1)
    #     near_goal_decel = (curr_dist < 4.0) & (speed < 1.0)
    #     reward[near_goal_decel] += 0.5 * (1.0 - speed[near_goal_decel])
    #     reward /= 10.0
    #     return reward

    # def _compute_reward(self, actions, reached, coll, wall_hit, prev_goal_dist, trunc):
    #     curr_dist = np.linalg.norm(self.goal - self.pos, axis=1)
    #
    #     # Calculate direction vectors
    #     r_vec = self.goal - self.pos
    #     dists = np.linalg.norm(r_vec, axis=1, keepdims=True) + 1e-8
    #     r_hat = r_vec / dists
    #
    #     # Radial velocity (movement directly toward goal)
    #     radial_vel = np.sum(self.vel * r_hat, axis=1)
    #
    #     # Tangential movement penalty (curved paths)
    #     v_radial = radial_vel[:, None] * r_hat
    #     v_tangential = self.vel - v_radial
    #     tang_speed = np.linalg.norm(v_tangential, axis=1)
    #
    #     # Progress rewards
    #     progress_reward = radial_vel * DT * PROGRESS_COEF
    #
    #     # Terminal rewards (per-agent)
    #     terminal = np.zeros(self.n, dtype=np.float32)
    #     terminal[reached] = R_GOAL
    #     terminal[coll] = R_COLL
    #     terminal[wall_hit] = R_WALL
    #     terminal[trunc & ~reached] = R_TIMEOUT
    #
    #     # Base rewards (per-agent)
    #     reward = (
    #             progress_reward +
    #             radial_vel * RADIAL_VEL_COEF -
    #             tang_speed * TANGENTIAL_PENALTY_COEF +
    #             terminal -
    #             ACTION_PENALTY_COEF * np.linalg.norm(actions, axis=1) -
    #             TIME_PENALTY
    #     )
    #
    #     # Alignment bonus near goal (per-agent)
    #     near_goal = (curr_dist < NEAR_GOAL_THRESHOLD) & (curr_dist > MIN_ALIGNMENT_DIST)
    #     if np.any(near_goal):
    #         v_norm = self.vel / (np.linalg.norm(self.vel, axis=1, keepdims=True) + 1e-8)
    #         cosine = np.sum(v_norm * r_hat, axis=1)
    #         reward[near_goal] += ALIGNMENT_BONUS_NEAR_GOAL_COEF * cosine[near_goal]
    #
    #     # Multi-drone avoidance bonus (per-agent)
    #     if self.n > 1:
    #         for i in range(self.n):
    #             # Calculate distance to other drones
    #             other_dists = np.linalg.norm(self.pos - self.pos[i], axis=1)
    #             other_dists[i] = np.inf  # Ignore self
    #             min_drone_dist = np.min(other_dists) - self.radii[i]
    #
    #             # Apply avoidance bonus
    #             if min_drone_dist < DRONE_AVOIDANCE_DIST:
    #                 # Normalized distance (0-1) within avoidance range
    #                 norm_dist = min_drone_dist / DRONE_AVOIDANCE_DIST
    #
    #                 # Reward increases with distance (max when at avoidance boundary)
    #                 avoidance_bonus = DRONE_AVOIDANCE_COEF * norm_dist
    #                 reward[i] += avoidance_bonus
    #
    #                 # Additional penalty if too close (safety critical)
    #                 if min_drone_dist < 1.5:
    #                     reward[i] -= 2.0 * (1.5 - min_drone_dist)
    #
    #     return reward


# def _compute_reward(self, actions, reached, coll, wall_hit, prev_goal_dist, trunc):
#     # ─── tunables ─────────────────────────────────────────────
#     SAFE_DIST = 12.0  # begin repelling here
#     DANG_DIST = 6.0  # heavy penalty inside
#     INV_COEF = 1.0  # inverse-square repulsion
#     COLL_PEN = 15.0  # softer collision penalty (no immediate done)
#     REACH_BONUS = 50.0  # reach goal bonus
#
#     # shaping
#     PROG_COEF = 4.0
#     STEP_COST = 0.02
#     ACT_PEN = 0.01
#     BRAKE_R = 12.0
#     BRAKE_COEF = 2.0
#     FUNNEL_R = 15.0
#     ZERO_PUSH = 0.1
#     AWAY_BONUS = 0.5  # smaller away bonus
#
#     eps = 1e-8
#     n = self.n
#
#     curr_d = np.linalg.norm(self.goal - self.pos, axis=1)
#     progress = prev_goal_dist - curr_d
#
#     # nearest-neighbor surface distance
#     rel_all = self.pos[:, None, :] - self.pos[None, :, :]
#     dists = np.linalg.norm(rel_all, axis=-1)
#     radii_sum = self.radii[None, :] + self.radii[:, None]
#     surf_mat = dists - radii_sum
#     np.fill_diagonal(surf_mat, np.inf)
#     nn_surf = surf_mat.min(axis=1)
#
#     rew = np.zeros(n, dtype=np.float32)
#     for i in range(n):
#         # ── terminal on actual trunc/timeout
#         if reached[i]:
#             rew[i] += REACH_BONUS
#             continue
#         if trunc[i] or wall_hit[i]:
#             rew[i] -= COLL_PEN
#             continue
#
#         # 1) progress + cost
#         rew[i] += PROG_COEF * progress[i]
#         rew[i] -= STEP_COST
#         rew[i] -= ACT_PEN * np.linalg.norm(actions[i])
#
#         # 2) goal align & speed-clear
#         g = self.goal[i] - self.pos[i]
#         dist = np.linalg.norm(g) + eps
#         ghat = g / dist
#         v = self.vel[i]
#         speed = np.linalg.norm(v)
#
#         adir = actions[i] / (np.linalg.norm(actions[i]) + eps)
#         rew[i] += np.dot(adir, ghat)
#         vdir = v / (speed + eps)
#         rew[i] += np.dot(vdir, ghat)
#
#         if dist > BRAKE_R and nn_surf[i] > SAFE_DIST:
#             rew[i] += 0.1 * speed
#
#         # 3) final-mile funnel + brake + stall-nudge
#         if dist < FUNNEL_R:
#             rew[i] += (1.0 - (dist / FUNNEL_R) ** 2)
#             rad_acc = np.dot(actions[i], ghat)
#             if dist < BRAKE_R and rad_acc < 0:
#                 rew[i] += BRAKE_COEF * (-rad_acc)
#             if speed < ZERO_PUSH:
#                 rew[i] -= ZERO_PUSH
#
#         # 4) tangential penalty
#         vrad = np.dot(v, ghat)
#         tang = v - vrad * ghat
#         rew[i] -= 0.5 * np.linalg.norm(tang)
#
#         # 5) gentle inverse‐square repulsion
#         d = nn_surf[i]
#         if d < SAFE_DIST:
#             φnow = INV_COEF / max(d, eps) ** 2
#             φsafe = INV_COEF / SAFE_DIST ** 2
#             rew[i] -= (φnow - φsafe)
#         if d < DANG_DIST:
#             rew[i] -= COLL_PEN * (1.0 - d / DANG_DIST)
#
#         # 6) small “away” bonus
#         j = surf_mat[i].argmin()
#         dir_ij = (self.pos[j] - self.pos[i]) / (dists[i, j] + eps)
#         toward = np.dot(v, dir_ij)
#         if toward < 0:
#             rew[i] += AWAY_BONUS * (-toward)
#
#         # 7) obstacle repulsion (same style)
#         if self.num_obstacles:
#             od = np.linalg.norm(self.obs_pos - self.pos[i], axis=1)
#             k = od.argmin()
#             so = od[k] - (self.radii[i] + self.obs_rad[k])
#             if so < SAFE_DIST:
#                 φo_now = INV_COEF / max(so, eps) ** 2
#                 φo_safe = INV_COEF / SAFE_DIST ** 2
#                 rew[i] -= (φo_now - φo_safe)
#             if so < DANG_DIST:
#                 rew[i] -= COLL_PEN * (1.0 - so / DANG_DIST)
#
#     return rew
# def _compute_reward(self, actions, reached, collided, wall_hit, prev_goal_dist, trunc):
#     R_GOAL = +150.0
#     R_COLLISION = -250.0
#     R_WALL = -80.0
#     R_TRUNC = -100.0
#     STEP_COST = -0.01
#     ACTION_COST = -0.005
#     PROGRESS_WEIGHT = 3.0
#     ALIGN_BONUS = 2.0
#     ZERO_SPEED_PENALTY = -3.0
#     SAFE_DIST = 8.0
#     REPULSION_COEF = 10.0
#     REPULSION_SOFT_RADIUS = 15.0
#     OBSTACLE_REPULSION_COEF = 8.0
#     OBSTACLE_REPULSION_RADIUS = 10.0
#
#     n = self.n
#     reward = np.zeros(n, dtype=np.float32)
#
#     for i in range(n):
#         if trunc[i]:
#             reward[i] += R_TRUNC
#
#         if reached[i]:
#             reward[i] += R_GOAL
#
#             # Bonus for stopping safely after reaching
#             if np.linalg.norm(self.vel[i]) < 0.3:
#                 reward[i] += 2.0
#
#         if collided[i]:
#             reward[i] += R_COLLISION
#
#         if wall_hit[i]:
#             reward[i] += R_WALL
#
#         # Dense goal shaping with tapering near other drones
#         curr_dist = np.linalg.norm(self.pos[i] - self.goal[i])
#         delta = prev_goal_dist[i] - curr_dist
#         if any(np.linalg.norm(self.pos[i] - self.pos[j]) < SAFE_DIST for j in range(n) if j != i):
#             delta *= 0.4
#         reward[i] += PROGRESS_WEIGHT * delta
#
#         # Heading alignment bonus
#         vel = self.vel[i]
#         dir_to_goal = self.goal[i] - self.pos[i]
#         goal_dist = np.linalg.norm(dir_to_goal)
#         if goal_dist > 1e-5 and np.linalg.norm(vel) > 0.1:
#             cosine_sim = np.dot(vel, dir_to_goal) / (np.linalg.norm(vel) * goal_dist)
#             reward[i] += ALIGN_BONUS * max(0.0, cosine_sim)
#
#         # Stall penalty
#         if goal_dist > 5.0 and np.linalg.norm(vel) < 0.3:
#             reward[i] += ZERO_SPEED_PENALTY
#
#         # Action penalty
#         reward[i] += ACTION_COST * np.linalg.norm(actions[i]) ** 2
#
#         # Bonus for moving safely in open space
#         if np.linalg.norm(self.vel[i]) > 0.5 and not any(
#                 np.linalg.norm(self.pos[i] - self.pos[j]) < 5.0 for j in range(n) if j != i
#         ):
#             reward[i] += 0.02
#
#         # Drone-to-drone repulsion and alignment bonus
#         for j in range(n):
#             if j != i:
#                 d = np.linalg.norm(self.pos[i] - self.pos[j])
#                 if d < REPULSION_SOFT_RADIUS:
#                     scale = (REPULSION_SOFT_RADIUS - d) / REPULSION_SOFT_RADIUS
#                     penalty = REPULSION_COEF * scale ** 2
#                     reward[i] -= penalty
#
#                 # Bonus for safe parallel motion
#                 if d < 10.0:
#                     vel_j = self.vel[j]
#                     cosine = np.dot(vel, vel_j) / (np.linalg.norm(vel) * np.linalg.norm(vel_j) + 1e-8)
#                     reward[i] += 2.0 * max(0.0, cosine)
#
#         # Drone-to-obstacle repulsion
#         for k in range(self.num_obstacles):
#             obs_vec = self.pos[i] - self.obs_pos[k]
#             obs_dist = np.linalg.norm(obs_vec)
#             rad = self.obs_rad[k]
#             buffer_dist = OBSTACLE_REPULSION_RADIUS + rad
#             if obs_dist < buffer_dist:
#                 repulse_strength = OBSTACLE_REPULSION_COEF * (buffer_dist - obs_dist) / buffer_dist
#                 reward[i] -= repulse_strength
#
#         # Step cost
#         reward[i] += STEP_COST
#
#         # Bonus for surviving without incident
#         if not collided[i] and not wall_hit[i] and not trunc[i]:
#             reward[i] += 5.0
#
#     return reward

#  100 -- 70
# def _compute_reward(self, actions, reached, coll, wall_hit, prev_goal_dist, trunc):
#     # ───── tunables ────────────────────────────────────────────────
#     SAFE_DIST = 12.0
#     DANGEROUS_DIST = 6.0
#     FUNNEL_R = 15.0
#     BRAKE_R = 9.0
#     ALIGN_R = 15.0
#     MIN_PUSH_SPEED = 0.5
#
#     # core coefficients
#     PROG_COEF = 5.0
#     STEP_COST = 0.01
#     ACTION_PEN = 0.005
#     ALIGN_COEF = 0.5
#     BRAKE_COEF = 2.0
#     REP_COEF = 5.0  # lighter soft-repulsion
#     COLLISION_PEN = 50.0  # moderate discrete penalty
#     REACH_BONUS = 150.0
#     ZERO_SPEED_PEN = 0.5
#
#     # continuous proximity cost (quadratic)
#     PROX_POW = 2.0  # (SAFE_DIST - d)^2
#
#     # active dodge bonuses
#     AWAY_DRONE_COEF = 2.0
#     AWAY_OBS_COEF = 1.0
#
#     eps = 1e-8
#     n = self.n
#
#     # distances & progress
#     curr_dist = np.linalg.norm(self.goal - self.pos, axis=1)
#     progress = prev_goal_dist - curr_dist
#
#     # compute nearest-neighbor surfaces
#     rel_all = self.pos[:, None, :] - self.pos[None, :, :]
#     dists = np.linalg.norm(rel_all, axis=-1)
#     radii_sum = self.radii[None, :] + self.radii[:, None]
#     surf_mat = dists - radii_sum
#     np.fill_diagonal(surf_mat, np.inf)
#     nn_surf = surf_mat.min(axis=1)
#
#     rew = np.zeros(n, dtype=np.float32)
#
#     for i in range(n):
#         # ── terminal cases ─────────────────────────────────────────
#         if reached[i]:
#             rew[i] += REACH_BONUS
#             continue
#         if coll[i] or wall_hit[i] or trunc[i]:
#             # only discrete penalty; we removed the giant slam so it won't freeze you
#             rew[i] -= COLLISION_PEN
#             continue
#
#         # 1) progress + tiny costs
#         rew[i] += PROG_COEF * progress[i]
#         rew[i] -= STEP_COST
#         rew[i] -= ACTION_PEN * np.linalg.norm(actions[i])
#
#         # 2) heading & velocity alignment
#         g = self.goal[i] - self.pos[i]
#         dist = np.linalg.norm(g) + eps
#         ghat = g / dist
#         v = self.vel[i]
#         speed = np.linalg.norm(v)
#
#         adir = actions[i] / (np.linalg.norm(actions[i]) + eps)
#         rew[i] += ALIGN_COEF * np.dot(adir, ghat)
#
#         vdir = v / (speed + eps)
#         rew[i] += ALIGN_COEF * np.dot(vdir, ghat)
#
#         # 2.5) speed-in-the-clear bonus
#         if dist > BRAKE_R and nn_surf[i] > SAFE_DIST:
#             rew[i] += 0.2 * speed
#
#         # 3) final‐mile funnel & braking
#         if dist < FUNNEL_R:
#             rew[i] += (1 - (dist / FUNNEL_R) ** 2)
#             rad_acc = np.dot(actions[i], ghat)
#             if dist < BRAKE_R and rad_acc < 0:
#                 rew[i] += BRAKE_COEF * (-rad_acc)
#             if speed < MIN_PUSH_SPEED:
#                 rew[i] -= ZERO_SPEED_PEN
#             if dist < ALIGN_R and speed > MIN_PUSH_SPEED * 3:
#                 rew[i] -= 2.0 * (speed - MIN_PUSH_SPEED * 3)
#
#         # 4) tangential penalty
#         vrad = np.dot(v, ghat)
#         tang = v - vrad * ghat
#         rew[i] -= 0.5 * np.linalg.norm(tang)
#
#         # 5) continuous proximity cost & soft repulsion
#         mn = nn_surf[i]
#         # quadratic cost inside SAFE_DIST
#         if mn < SAFE_DIST:
#             rew[i] -= 0.5 * ((SAFE_DIST - mn) / SAFE_DIST) ** PROX_POW
#         # linear soft-repel
#         if mn < SAFE_DIST:
#             rep_scale = np.clip(dist / FUNNEL_R, 0, 1)
#             rew[i] -= rep_scale * REP_COEF * (SAFE_DIST - mn) / SAFE_DIST
#
#         # 5b) active dodge away from drone
#         j = np.argmin(surf_mat[i])
#         dir_ij = (self.pos[i] - self.pos[j]) / (dists[i, j] + eps)
#         towards_j = np.dot(v, dir_ij)
#         if towards_j < 0:
#             rew[i] += AWAY_DRONE_COEF * (-towards_j)
#
#         # 6) obstacle repulsion & dodge
#         if self.num_obstacles > 0:
#             odists = np.linalg.norm(self.obs_pos - self.pos[i], axis=1)
#             k = np.argmin(odists)
#             surf_o = odists[k] - (self.radii[i] + self.obs_rad[k])
#
#             # continuous cost
#             if surf_o < SAFE_DIST:
#                 rew[i] -= 0.3 * ((SAFE_DIST - surf_o) / SAFE_DIST) ** PROX_POW
#             # soft-repel
#             if surf_o < SAFE_DIST:
#                 obs_scale = np.clip(dist / FUNNEL_R, 0, 1)
#                 rew[i] -= obs_scale * REP_COEF * (SAFE_DIST - surf_o) / SAFE_DIST
#             # active dodge from obstacle
#             dir_io = (self.pos[i] - self.obs_pos[k]) / (odists[k] + eps)
#             towards_o = np.dot(v, dir_io)
#             if towards_o < 0:
#                 rew[i] += AWAY_OBS_COEF * (-towards_o)
#
#     return rew