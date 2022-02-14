import os
import numpy as np
import pandas


def quick_get_success():
    def get_env_num(x):
        return 250

    dir = "/home/liquanyi/iclr_exp/iwr_eval_results"
    success_env_num = {}
    success_env = {}
    for file in sorted(os.listdir(dir), key=lambda x: get_env_num(x)):
        if "tmp" not in file:
            with open("{}/{}".format(dir, file), "r") as f:
                record = f.readlines()
            # print("Evaluating {}".format(file))
            env_num = get_env_num(file)
            if env_num not in success_env_num.keys():
                success_env_num[env_num] = []
            success = 0
            for i in record[1:]:
                env_index = int(i.split(",")[0])
                if env_index not in success_env.keys():
                    success_env[env_index] = {
                        "BATCH_PER_ITER": 0,
                        "F": 0,
                    }
                if "True" in i:
                    success += 1
                    success_env[env_index]["BATCH_PER_ITER"] += 1
                else:
                    success_env[env_index]["F"] += 1
            success_env_num[env_num].append(success)
    for env_num in success_env_num.keys():
        success_env_num[env_num] = sum(success_env_num[env_num]) / len(success_env_num[env_num])
    for env_index in success_env.keys():
        success_env[env_index] = round(
            success_env[env_index]["BATCH_PER_ITER"] / (
                    success_env[env_index]["BATCH_PER_ITER"] + success_env[env_index]["F"]), 4)
    print(success_env_num)
    print(success_env)

def parse_eval_res(path):
    with open(path, "r") as eval_file:
        df = pandas.read_csv(eval_file)
    for name in ["episode_reward", "episode_cost"]:
        print("{} mean: {},{} std: {}".format(name, np.mean(df[name]), name, np.std(df[name])))
    print("success: {}".format(np.sum(df["succZess"])/len(df["success"])))

if __name__ == "__main__":
    # quick_get_success()
    # parse_eval_res("/home/liquanyi/iclr_exp/iwr_eval_results/iwr.csv")
    parse_eval_res("/home/liquanyi/iclr_exp/hg_dagger_eval_results/hg_dagger.csv")