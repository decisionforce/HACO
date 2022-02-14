import argparse
import copy
import os
import os.path as osp
import time
from core.haco_env import HACOEnv
import numpy as np
import pandas as pd
from drivingforce.human_in_the_loop.common import baseline_eval_config
from drivingforce.human_in_the_loop.eval.get_policy_function import VisionPolicyFunction
from drivingforce.human_in_the_loop.eval.utils import pretty_print, RecorderEnv
from drivingforce.train.utils import initialize_ray

EVAL_ENV_START = 1000


def evaluate_once(
        ckpt_path,
        ckpt_name,
        ckpt_index,
        folder_name,
        # env_num,
        # start_seed,
        # num_episodes=10,
        use_render=False,
        num_ep_in_one_env=5,
        total_env_num=20,
):
    # ===== Evaluate populations =====
    os.makedirs("evaluate_results", exist_ok=True)
    saved_results = []

    # Setup policy
    try:
        policy_function = VisionPolicyFunction(ckpt_path)
    except FileNotFoundError:
        print("We failed to load: ", ckpt_path)
        return None

    os.makedirs(folder_name, exist_ok=True)

    # Setup environment
    env = make_env(use_render)
    try:

        start = time.time()
        last_time = time.time()
        ep_count = 0
        step_count = 0
        ep_times = []

        env_index = 0
        o = env.reset(force_seed=EVAL_ENV_START + env_index)

        num_ep_in = 0

        while True:
            # INPUT: [batch_size, obs_dim] or [obs_dim, ] array.
            # OUTPUT: [batch_size, act_dim] !! This is important!
            action = policy_function(o)["default_policy"]

            # Step the environment
            o, r, d, info = env.step(action)
            step_count += 1

            if use_render:
                env.render()

            # Reset the environment
            if d or (step_count >= 2000):

                print(
                    "Env {}, Num episodes: {} ({}), Num steps in this episode: {} (Ep time {:.2f}, "
                    "Total time {:.2f}). Ckpt: {}".format(
                        env_index, num_ep_in, ep_count, step_count,
                        np.mean(ep_times), time.time() - start, ckpt_path
                    )
                )

                policy_function.reset()
                step_count = 0
                ep_count += 1
                num_ep_in += 1
                env_id_recorded = EVAL_ENV_START + env_index
                num_ep_in_recorded = num_ep_in
                if num_ep_in >= num_ep_in_one_env:
                    env_index = min(env_index + 1, total_env_num - 1)
                    num_ep_in = 0

                o = env.reset(force_seed=EVAL_ENV_START + env_index)

                ep_times.append(time.time() - last_time)
                last_time = time.time()

                print("Finish {} episodes with {:.3f} s!".format(ep_count, time.time() - start))
                res = env.get_episode_result()
                res["episode"] = ep_count
                res["ckpt_index"] = ckpt_index
                res["env_id"] = env_id_recorded
                res["num_ep_in_one_env"] = num_ep_in_recorded
                saved_results.append(res)
                df = pd.DataFrame(saved_results)
                print(pretty_print(res))

                path = "{}/{}_tmp.csv".format(folder_name, ckpt_name)
                print("Backup data is saved at: ", path)
                df.to_csv(path)

                if env_index >= total_env_num - 1:
                    break

    except Exception as e:
        raise e
    finally:
        env.close()
        del policy_function

    df = pd.DataFrame(saved_results)
    print("===== Result =====")
    print("Checkpoint {} results (len {}): \n{}".format(ckpt_name, len(df), {k: round(df[k].mean(), 3) for k in df}))
    print("===== Result =====")
    path = "{}/{}.csv".format(folder_name, ckpt_name)
    print("Final data is saved at: ", path)
    df.to_csv(path)
    df["model_name"] = ckpt_name
    return df


def make_env(use_render=False):
    config = copy.deepcopy(baseline_eval_config)

    if use_render:
        config["use_render"] = True
        config["disable_model_compression"] = True

    env = HACOEnv(config, eval=True, port=10000)
    return RecorderEnv(env)


if __name__ == '__main__':
    initialize_ray(num_gpus=1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=".\EGPO_CARLA_211201-150058\HACO_HACOEnv_710f0_00000_0_seed=0_2021-12-01_15-01-00", type=str)
    parser.add_argument("--folder", default="carla_eval", type=str)
    parser.add_argument("--start_ckpt", default=1, type=int)
    parser.add_argument("--num_ckpt", type=int, default=85)
    parser.add_argument("--num_ep_in_one_env", type=int, default=2)
    parser.add_argument("--total_env_num", type=int, default=50)
    parser.add_argument("--skip", type=int, default=2)
    args = parser.parse_args()

    use_render = False

    trial_path = args.path

    start_ckpt = args.start_ckpt
    num_ckpt = args.num_ckpt

    for ckpt_index in reversed(range(start_ckpt, start_ckpt + num_ckpt, args.skip)):
        ckpt_path = osp.join(trial_path, "checkpoint_{}".format(ckpt_index), "checkpoint-{}".format(ckpt_index))
        if not osp.exists(ckpt_path):
            print("=====\nWe can't find checkpoint {}\n=====".format(ckpt_path))
            continue

        print("===== Start evaluating checkpoint {}. Will be saved at {} =====".format(ckpt_index, args.folder))

        ret = evaluate_once(
            ckpt_path=ckpt_path,
            ckpt_name="checkpoint_{}".format(ckpt_index),
            ckpt_index=ckpt_index,
            folder_name=args.folder,
            # env_num=env_num,
            # start_seed=start_seed,
            # num_episodes=num_episodes,
            use_render=use_render,
            num_ep_in_one_env=1,
            total_env_num=100,
        )
        if ret is None:
            print("We failed to evaluate.")
        else:
            print("\n\n\n Finish evaluation. \n\n\n")
