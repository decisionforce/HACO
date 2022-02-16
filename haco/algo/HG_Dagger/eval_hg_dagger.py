import copy
import os
import time

import numpy as np
import pandas as pd

from haco.utils.human_in_the_loop_env import HumanInTheLoopEnv
from algo.HG_Dagger.model import Ensemble
from haco.utils.config import baseline_eval_config
from haco.utils.eval_IWR_HG_Dagger import pretty_print, RecorderEnv

EVAL_ENV_START = 1000


def evaluate_once(
        ckpt_path,
        folder_name,
        exp_name,
        use_render=False,
        num_ep_in_one_env=5,
        total_env_num=50,
):
    # ===== Evaluate populations =====
    # Setup environment
    ckpt_name = exp_name
    env = make_env(use_render)

    saved_results = []

    # Setup policy
    model = Ensemble(env.observation_space.shape, env.action_space.shape, "cpu")
    model.load(ckpt_path)

    def policy_function(obs):
        return model.act(obs)

    os.makedirs(folder_name, exist_ok=True)

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
            action = policy_function(o)

            # Step the environment
            o, r, d, info = env.step(action)
            step_count += 1

            if use_render:
                env.render()

            # Reset the environment
            if d or (step_count >= 1500):

                print(
                    "Env {}, Num episodes: {} ({}), Num steps in this episode: {} (Ep time {:.2f}, "
                    "Total time {:.2f}). Ckpt: {}".format(
                        env_index, num_ep_in, ep_count, step_count,
                        np.mean(ep_times), time.time() - start, ckpt_path
                    )
                )

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

    df = pd.DataFrame(saved_results)
    print("===== Result =====")
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

    env = HumanInTheLoopEnv(config)
    return RecorderEnv(env)


if __name__ == "__main__":
    ckpt_path = None
    evaluate_once(ckpt_path, "hg_dagger_model_3_eval_results", "hg_dagger")
