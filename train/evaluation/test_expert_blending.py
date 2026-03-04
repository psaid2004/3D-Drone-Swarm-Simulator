"""
Run the ExpertPolicy in the 3-D drone env and see whether it ever reaches the target.

python -m train.evaluation.test_expert_blending
"""

import numpy as np
from config.defaults import DEFAULT_NUM_DRONES, DEFAULT_LIDAR
from wrappers.pz_parallel_3d import env as make_env
from train.sb3_sac_training import ExpertPolicy

NUM_EPISODES = 10
MAX_STEPS = 1000
WORLD_SCALE = 5
ACTIVE_DRONES = 1
NUM_OBSTACLES = 10

def run():
    expert = ExpertPolicy()

    raw = make_env(
        num_drones=DEFAULT_NUM_DRONES,
        active_drones=ACTIVE_DRONES,
        num_obstacles=NUM_OBSTACLES,
        max_steps=MAX_STEPS,
        lidar=DEFAULT_LIDAR,
        render_mode=None,
        world_scale=WORLD_SCALE,
    )

    for ep in range(NUM_EPISODES):
        obs, infos = raw.reset()
        total_rewards = {a: 0.0 for a in raw.agents}
        reached = {a: False for a in raw.agents}

        for step in range(MAX_STEPS):
            actions = {}
            for agent_id, ob in obs.items():
                actions[agent_id] = expert.expert_action(ob)

            obs, rewards, terms, truncs, infos = raw.step(actions)

            for a in raw.agents:
                total_rewards[a] += rewards.get(a, 0.0)
                # check if the env signals "reached" via info or termination
                if infos.get(a, {}).get("reached", False):
                    reached[a] = True

            # end episode if all done
            if all(terms.get(a, False) or truncs.get(a, False) for a in raw.agents):
                break

        print(f"\nEpisode {ep+1}/{NUM_EPISODES}  (steps={step+1})")
        for a in raw.agents:
            tag = "REACHED" if reached[a] else "missed"
            print(f"  {a}: reward={total_rewards[a]:+.2f}  [{tag}]")

        # also print final position vs target if available in info
        for a in raw.agents:
            info = infos.get(a, {})
            if "pos" in info and "target" in info:
                pos = np.array(info["pos"])
                tgt = np.array(info["target"])
                dist = np.linalg.norm(pos - tgt)
                print(f"  {a}: final_pos={pos}, target={tgt}, dist={dist:.2f}")

    raw.close()
    print("\nDone.")


if __name__ == "__main__":
    run()