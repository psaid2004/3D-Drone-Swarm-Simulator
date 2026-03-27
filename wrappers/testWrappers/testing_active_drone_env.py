import time
import numpy as np
from wrappers.env_3d import DroneSwarm3D


def safe_unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.zeros_like(v, dtype=np.float32)
    return (v / norm).astype(np.float32)


def make_policy_actions(obs_dict: dict[str, np.ndarray], active_drones: int) -> dict[str, np.ndarray]:
    """
    Build actions for all visible agents.
    Active drones move toward target.
    Inactive padded drones are explicitly frozen.
    """
    action_dict: dict[str, np.ndarray] = {}

    for i, (drone, obs) in enumerate(obs_dict.items()):
        if i >= active_drones:
            action_dict[drone] = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
            continue

        to_target = obs[6:9]
        to_centroid = obs[12:15]

        dir_vec = 0.8 * safe_unit(to_target) + 0.2 * safe_unit(to_centroid)
        dir_vec = safe_unit(dir_vec)

        speed = np.clip(np.linalg.norm(to_target) / 5.0, 0.05, 1.0)
        u3 = speed * 2.0 - 1.0

        action_dict[drone] = np.array(
            [dir_vec[0], dir_vec[1], dir_vec[2], u3],
            dtype=np.float32,
        )

    return action_dict


def print_step_summary(env, step: int, obs_dict, reward_dict, info_dict, max_lines: int = 12) -> None:
    print(f"\n===== STEP {step} =====")
    shown = 0
    for drone in obs_dict:
        if shown >= max_lines:
            break

        pos = obs_dict[drone][:3]

        i = int(drone.split("_")[1])
        if i < env.max_total_num_drones:
            real_pos = env._env._getDroneStateVector(i)[:3]
        else:
            real_pos = np.zeros(3)
        r = reward_dict.get(drone, 0.0)
        inf = info_dict.get(drone, {})

        print(
            f"{drone:10s} | "
            f"pos=({real_pos[0]:7.2f},{real_pos[1]:7.2f},{real_pos[2]:7.2f}) | "
            f"reward={r:8.3f} | "
            f"active={str(inf.get('active', False)):5s} | "
            f"reached={str(inf.get('reached', False)):5s} | "
            f"collided={str(inf.get('collided', False)):5s}"
        )
        shown += 1


def check_padded_drones_are_inert(
    obs_dict: dict[str, np.ndarray],
    reward_dict: dict[str, float],
    term_dict: dict[str, bool],
    trunc_dict: dict[str, bool],
    info_dict: dict[str, dict],
    active_drones: int,
) -> None:
    """
    Assert that padded drones behave like neutral masked slots.
    """
    agent_names = list(obs_dict.keys())
    padded = agent_names[active_drones:]

    for drone in padded:
        obs = obs_dict[drone]
        reward = reward_dict.get(drone, None)
        term = term_dict.get(drone, None)
        trunc = trunc_dict.get(drone, None)
        info = info_dict.get(drone, {})

        if not np.allclose(obs, 0.0):
            raise AssertionError(f"{drone} is padded but observation is not all zeros.")

        if reward != 0.0:
            raise AssertionError(f"{drone} is padded but reward is {reward}, expected 0.0.")

        if term not in (False, None):
            raise AssertionError(f"{drone} is padded but terminated={term}, expected False.")

        if trunc not in (False, None):
            raise AssertionError(f"{drone} is padded but truncated={trunc}, expected False.")

        if info.get("active", True) is not False:
            raise AssertionError(f"{drone} is padded but info['active'] is not False.")


def run_episode(active_drones: int, max_drones: int, n_steps: int = 300, render: bool = True) -> None:
    print(f"\n=== RUN EPISODE | active_drones={active_drones}, max_drones={max_drones} ===")

    env = DroneSwarm3D(
        world_scale=4.0,
        num_drones_per_swarm=active_drones,
        num_static_obstacles=0,
        num_dynamic_obstacles=0,
        max_steps=n_steps,
        num_swarms=1,
        lidar=True,
        render_mode="human" if render else None,
        max_drones_per_swarm=max_drones,
    )

    obs_dict, info = env.reset()

    print(f"len(possible_agents) = {len(env.possible_agents)}")
    print(f"total_num_drones     = {env.total_num_drones}")
    print(f"max_total_num_drones = {env.max_total_num_drones}")
    print(f"len(obs_dict)        = {len(obs_dict)}")

    assert len(env.possible_agents) == max_drones, "possible_agents should equal fixed max capacity"
    assert env.total_num_drones == active_drones, "active drone count mismatch"
    assert env.max_total_num_drones == max_drones, "max drone count mismatch"
    assert len(obs_dict) == max_drones, "obs_dict should contain all fixed-capacity agents"

    # Initial padded-agent sanity check
    initial_rewards = {a: 0.0 for a in obs_dict}
    initial_terms = {a: False for a in obs_dict}
    initial_truncs = {a: False for a in obs_dict}
    initial_infos = {a: {"active": i < active_drones} for i, a in enumerate(obs_dict.keys())}
    check_padded_drones_are_inert(
        obs_dict, initial_rewards, initial_terms, initial_truncs, initial_infos, active_drones
    )

    for step in range(n_steps):
        action_dict = make_policy_actions(obs_dict, active_drones)
        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)

        check_padded_drones_are_inert(
            obs_dict, reward_dict, term_dict, trunc_dict, info_dict, active_drones
        )

        if step % 50 == 0:
            print_step_summary(env, step, obs_dict, reward_dict, info_dict)

        # For active drones only
        active_agent_names = list(obs_dict.keys())[:active_drones]
        if all(term_dict[a] or trunc_dict[a] for a in active_agent_names):
            print(f"Episode ended at step {step}")
            print_step_summary(env, step, obs_dict, reward_dict, info_dict)
            break

        if render:
            env.render()
            time.sleep(1 / 60)

    env.close()
    print("[OK] episode run completed")


def run_reset_test(active_drones: int, max_drones: int, n_steps_before_reset: int = 50) -> None:
    print(f"\n=== RESET TEST | active_drones={active_drones}, max_drones={max_drones} ===")

    env = DroneSwarm3D(
        world_scale=4.0,
        num_drones_per_swarm=active_drones,
        num_static_obstacles=0,
        num_dynamic_obstacles=0,
        max_steps=300,
        num_swarms=1,
        lidar=True,
        render_mode=None,
        max_drones_per_swarm=max_drones,
    )

    obs_dict, _ = env.reset()

    for _ in range(n_steps_before_reset):
        action_dict = make_policy_actions(obs_dict, active_drones)
        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)

    obs_dict2, info2 = env.reset()

    assert len(obs_dict2) == max_drones, "reset changed obs_dict size"
    assert len(env.possible_agents) == max_drones, "reset changed possible_agents size"
    assert env.total_num_drones == active_drones, "reset changed active drone count"
    assert env.max_total_num_drones == max_drones, "reset changed max drone count"

    check_padded_drones_are_inert(
        obs_dict2,
        {a: 0.0 for a in obs_dict2},
        {a: False for a in obs_dict2},
        {a: False for a in obs_dict2},
        {a: {"active": i < active_drones} for i, a in enumerate(obs_dict2.keys())},
        active_drones,
    )

    env.close()
    print("[OK] reset test passed")


def run_stage_transition_test(max_drones: int = 10) -> None:
    """
    Mimics curriculum stage transition without SB3:
    stage 0 -> 1 active drone
    stage 1 -> 2 active drones
    both with same fixed max capacity
    """
    print(f"\n=== STAGE TRANSITION TEST | max_drones={max_drones} ===")

    # Stage 0
    env0 = DroneSwarm3D(
        world_scale=4.0,
        num_drones_per_swarm=1,
        num_static_obstacles=0,
        num_dynamic_obstacles=0,
        max_steps=150,
        num_swarms=1,
        lidar=True,
        render_mode=None,
        max_drones_per_swarm=max_drones,
    )

    obs0, _ = env0.reset()
    for _ in range(40):
        a0 = make_policy_actions(obs0, active_drones=1)
        obs0, rew0, term0, trunc0, info0 = env0.step(a0)
        check_padded_drones_are_inert(obs0, rew0, term0, trunc0, info0, active_drones=1)

    env0.close()
    print("[OK] stage 0 closed cleanly")

    # Stage 1
    env1 = DroneSwarm3D(
        world_scale=4.0,
        num_drones_per_swarm=2,
        num_static_obstacles=0,
        num_dynamic_obstacles=0,
        max_steps=150,
        num_swarms=1,
        lidar=True,
        render_mode=None,
        max_drones_per_swarm=max_drones,
    )

    obs1, _ = env1.reset()
    assert len(obs1) == max_drones, "stage 1 obs size changed"
    assert env1.total_num_drones == 2, "stage 1 active drone count wrong"
    assert env1.max_total_num_drones == max_drones, "stage 1 max drone count wrong"

    for _ in range(40):
        a1 = make_policy_actions(obs1, active_drones=2)
        obs1, rew1, term1, trunc1, info1 = env1.step(a1)
        check_padded_drones_are_inert(obs1, rew1, term1, trunc1, info1, active_drones=2)

    env1.close()
    print("[OK] stage 1 run completed")
    print("[OK] stage transition test passed")


def main():
    # 1) visual smoke test
    run_episode(active_drones=4, max_drones=10, n_steps=8000, render=True)

    # 2) reset stability
    run_reset_test(active_drones=1, max_drones=10, n_steps_before_reset=50)

    # 3) stage-like change in active drone count
    run_stage_transition_test(max_drones=10)

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()