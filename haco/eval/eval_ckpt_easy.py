import argparse
import os
import time

import numpy as np
import pandas as pd
from drivingforce.real_data_generalization.eval_ckpt.get_policy_function import PolicyFunction
from drivingforce.real_data_generalization.eval_ckpt.utils import pretty_print, RecorderEnv
from metadrive import MetaDriveEnv
# from metadrive.envs.argoverse_env import ArgoverseGeneralizationEnv


def evaluate_once(env_num, start_seed, num_episodes=10, use_render=False):
    # ===== Evaluate populations =====
    os.makedirs("evaluate_results", exist_ok=True)
    saved_results = []

    # Setup policy
    try:
        policy_function = PolicyFunction(env_num=env_num, start_seed=start_seed)
    except FileNotFoundError:
        print("We failed to load: ", env_num, start_seed)
        return None

    # Setup environment
    env = make_env(use_render)
    try:
        o = env.reset(force_seed=0)
        d = {"__all__": False}
        start = time.time()
        last_time = time.time()
        ep_count = 0
        step_count = 0
        ep_times = []
        while True:

            # INPUT: [batch_size, obs_dim] or [obs_dim, ] array.
            # OUTPUT: [batch_size, act_dim] !! This is important!
            action = policy_function(o)[0]

            # Step the environment
            o, r, d, info = env.step(action)
            step_count += 1

            if use_render:
                env.render()

            if step_count % 100 == 0:
                print(
                    "Evaluating {}, Num episodes: {}, Num steps in this episode: {} (Ep time {:.2f}, "
                    "Total time {:.2f})".format(
                        policy_function.model_name, ep_count, step_count, np.mean(ep_times), time.time() - start
                    )
                )

            # Reset the environment
            if d:
                policy_function.reset()

                step_count = 0
                ep_count += 1
                o = env.reset(force_seed=ep_count)

                ep_times.append(time.time() - last_time)
                last_time = time.time()

                print("Finish {} episodes with {:.3f} s!".format(ep_count, time.time() - start))
                res = env.get_episode_result()
                res["episode"] = ep_count
                saved_results.append(res)
                df = pd.DataFrame(saved_results)
                print(pretty_print(res))

                path = "evaluate_results/{}_tmp.csv".format(policy_function.model_name)
                print("Backup data is saved at: ", path)
                df.to_csv(path)

                d = {"__all__": False}
                if ep_count >= num_episodes:
                    break
    except Exception as e:
        raise e
    finally:
        env.close()

    df = pd.DataFrame(saved_results)
    path = "evaluate_results/{}.csv".format(policy_function.model_name)
    print("Final data is saved at: ", path)
    df.to_csv(path)
    df["model_name"] = policy_function.model_name
    return df


def make_env(use_render=False):
    # env = MetaDriveEnv({
        # "random_agent_model": True,
        # "random_lane_width": True,
        # "random_lane_num": True,

        # "disable_model_compression": use_render,
        # "use_render": use_render,

        # "vehicle_config": {"lane_line_detector": {"disable_lane_line_detection": True}}
    # })
    env = ArgoverseGeneralizationEnv({
        "environment_num": 74,
        "mode": "all",
        "source": "tracking",
        "start_seed": 0,
        "random_agent_model": True,
        # "random_lane_width": True,
        # "random_lane_num": True,

        # "disable_model_compression": use_render,
        # "use_render": use_render,

        "vehicle_config": {"lane_line_detector": {"disable_lane_line_detection": True}},
    })

    env = RecorderEnv(env)
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_num", required=True, type=int)
    parser.add_argument("--start_seed", required=True, type=int)
    args = parser.parse_args()
    # name = args.name

    use_render = False

    # Information about the checkpoint
    env_num = args.env_num
    start_seed = args.start_seed

    num_episodes = 70

    ret = evaluate_once(
        env_num=env_num,
        start_seed=start_seed,
        num_episodes=num_episodes,
        use_render=use_render
    )
    if ret is None:
        print("We failed to evaluate.")
    else:
        print("\n\n\n Finish evaluation. \n\n\n")
