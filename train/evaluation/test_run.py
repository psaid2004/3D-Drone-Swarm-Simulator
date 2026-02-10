from stable_baselines3 import PPO

# from train.sb_ppo_training import CURRICULUM
from wrappers.pz_parallel import env as make_env

NUM_DRONES = 2
NUM_OBSTACLES = 4
MODEL_N = 0
N_STEPS = 510
LIDAR = True

MODEL = f"ppo_{MODEL_N}_{NUM_DRONES}_{NUM_OBSTACLES}"
MODEL_PATH = f"../models/{MODEL}.zip"
# MODEL = "ppo_drones_0_2_4/ppo__2272416_steps"
# MODEL_PATH = f"../models/ppo_drones_50_2_0/ppo__680000_steps.zip"


def print_rewards(reward_dictionary):
    print("\nReward system:\n")
    for k in reward_dictionary.keys():
        print(f'\n{k} - {reward_dictionary[k]}')
    print(reward_dictionary)


# 1) load your policy
def test_run(n_drones=NUM_DRONES, n_obstacles=NUM_OBSTACLES, n_steps=N_STEPS, model_path=MODEL_PATH):
    model = PPO.load(model_path)  # adjust path if needed

    # 2) make a ParallelEnv with human rendering
    pz_env = make_env(num_drones=n_drones, render_mode="human", num_obstacles=n_obstacles, max_steps=n_steps,
                      lidar=LIDAR)

    # 3) reset
    obs_dict, info = pz_env.reset()
    sims_number = 10
    # 4) episode loop
    while sims_number != 0:
        obs_dict, _ = pz_env.reset()
        j = 0
        while j < n_steps:
            # build actions for currently alive agents
            action_dict = {a: model.predict(o, deterministic=True)[0]
                           for a, o in obs_dict.items()}

            obs_dict, reward_dict, term_dict, trunc_dict, info = pz_env.step(action_dict)

            pz_env.render()
            # print(sims_number)
            print(info)
            # if j >= 60:
            #     time.sleep(1)
            # print(obs_dict)
            # episode is over when the live agent list is empty
            if len(pz_env.agents) == 0:  # <-- this is the correct check
                print("Episode ended, final info:", info)
                break
            j += 1
        sims_number -= 1

    # 5) clean up
    pz_env.close()


# def test_full_curriculum():
#     for stage, (num_drones, num_obstacles, timesteps) in enumerate(CURRICULUM):
#         test_run(num_drones, num_obstacles, 300, f"../stage_{stage}_model")
#
#
# def test_curriculum(stage=1):
#     num_drones, num_obstacles, timesteps = CURRICULUM[stage - 1]
#     test_run(num_drones, num_obstacles, 300, f"../stage_{stage}_model")


if __name__ == '__main__':
    # test_curriculum(3)
    test_run()
