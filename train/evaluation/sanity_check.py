import numpy as np

from envs.drone_env import DroneWorldEnv

# ---------------- Tunables -------------------------------------------
N_EPISODES = 100
K_P = 2.0  # position gain
K_D = 0.5  # damping gain
MAX_V = 5.0
MAX_A = 2.0  # maximum acceleration
MAX_STEPS = 400
SUCCESS_DIST = 1.5


# ---------------------------------------------------------------------

def run_n_episodes(num_ep=1):
    for i in range(num_ep):
        print(f"ENV Details run number: {i}\n\n")
        env = DroneWorldEnv(num_drones=1,
                            num_obstacles=25,
                            lidar=False,
                            max_steps=MAX_STEPS,
                            render_mode="human")  # Set to "human" to watch

        succ, wall_hit, coll, lens, rets = 0, 0, 0, [], []
        timeouts = 0

        for ep in range(N_EPISODES):
            obs, _ = env.reset()
            EP_RET = 0.0
            reached = False

            for t in range(MAX_STEPS):
                pos = env.pos[0]
                goal = env.goal[0]
                vel = env.vel[0]
                r_vec = goal - pos
                dist = np.linalg.norm(r_vec) + 1e-8
                r_hat = r_vec / dist  # unit radial vector
                env.render()
                # Calculate current speed
                current_speed = np.linalg.norm(vel)

                # Velocity-based adaptive control
                if dist > 10.0:
                    # Full speed when far from goal
                    target_speed = min(MAX_V, dist * 0.5)  # Dynamic speed target
                elif dist > 3.0:
                    # Moderate speed in mid-range
                    target_speed = min(MAX_V * 0.7, dist * 0.3)
                else:
                    # Slow approach for final meters
                    target_speed = min(0.5, dist * 0.2)

                # Velocity error
                v_error = target_speed - current_speed

                # Direction control
                desired_direction = r_hat

                # Velocity vector adjustment
                v_rad = np.dot(vel, r_hat) * r_hat  # radial component
                v_tan = vel - v_rad  # tangential component

                # --- radial PD toward goal ------------------------------------------
                a_rad = K_P * r_vec - K_D * v_rad

                # --- tangential damping ---------------------------------------------
                a_tan = -1.0 * v_tan

                # --- speed control component ----------------------------------------
                a_speed = v_error * r_hat

                # Combine acceleration components
                desired_acc = a_rad + a_tan + a_speed

                # Clip to maximum acceleration
                action = np.clip(desired_acc, -MAX_A, MAX_A)[None, :]

                # Take step
                obs, rew, done, trunc, info = env.step(action)
                # print(info)
                print(obs[0])
                EP_RET += rew[0]

                # Check if reached
                if env._check_reached()[0]:
                    reached = True

                if done[0] or trunc[0] or reached:
                    lens.append(t + 1)
                    rets.append(EP_RET)

                    if info["drone_0"]["wall_hit"]:
                        wall_hit += 1
                    elif reached:
                        succ += 1
                    elif trunc[0]:
                        timeouts += 1
                    else:
                        coll += 1
                    break

        print(f"episodes: {N_EPISODES}")
        print(f"  success rate : {succ / N_EPISODES:.2%}")
        print(f"  timeout rate : {timeouts / N_EPISODES:.2%}")
        print(f"  collision rate: {coll / N_EPISODES:.2%}")
        print(f"  wall-hit rate : {wall_hit / N_EPISODES:.2%}")
        print(f"  mean steps    : {np.mean(lens):.1f}")
        # print(f"  mean return   :
        env.close()


if __name__ == '__main__':
    run_n_episodes(10)
    # env = DroneWorldEnv(num_drones=1, num_obstacles=0, render_mode="human")
    # obs, _ = env.reset()
    # MAX_A = 2.0
    # success_count = 0
    #
    # for _ in range(1000):
    #     # Simple proportional controller
    #     goal_dir = env.goal[0] - env.pos[0]
    #     dist = np.linalg.norm(goal_dir)
    #     action = (goal_dir / (dist + 1e-8)) * MAX_A
    #     env.render()
    #     _, _, done, trunc, _ = env.step([action])
    #
    #     if done[0]:
    #         success_count += 1
    #         env.reset()
    #
    #     if trunc[0]:
    #         env.reset()
    #
    # print(f"Success rate: {success_count / 10}%")
