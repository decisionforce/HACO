"""
This script is used to remove the optimizer state in the checkpoint. So that we can compress 2/3 of the checkpoint size.
This script is put here for reference only. In formal release, the original checkpoint file will be removed so
this script will become not runnable.
"""
import os
import os.path as osp
import pickle

import numpy as np

ckpt_path = osp.join(osp.dirname(__file__), "checkpoint_417/checkpoint-417")
# "/Users/pengzhenghao/PycharmProjects/drivingforce/results/neurips21/checkpoints_of_generalization/PPO_PGDriveEnv_f13d1_00000_0_environment_num=1,start_seed=5000,num_lasers=0,seed=0_2021-08-23_00-06-43/checkpoint_500"
root_path = "/Users/pengzhenghao/PycharmProjects/drivingforce/results/neurips21/checkpoints_of_generalization/"


def remove_useless_state(ckpt_path, save_path):
    remove_value_network = True
    with open(ckpt_path, "rb") as f:
        data = f.read()
    unpickled = pickle.loads(data)
    worker = pickle.loads(unpickled.pop("worker"))
    if "_optimizer_variables" in worker["state"]["default_policy"]:
        worker["state"]["default_policy"].pop("_optimizer_variables")
    weights = worker["state"]["default_policy"]
    if remove_value_network:
        weights = {k: v for k, v in weights.items() if "value" not in k}
    np.savez_compressed(save_path, **weights)
    print("Numpy agent weight is saved at: {}!".format(save_path))


if __name__ == '__main__':

    for trial_path in os.listdir(root_path):
        trial_path = osp.join(root_path, trial_path, "checkpoint_500", "checkpoint-500")
        assert osp.exists(trial_path)
        env_num = eval(trial_path.split("environment_num=")[1].split(",")[0])
        start_seed = eval(trial_path.split("start_seed=")[1].split(",")[0])
        save_path = osp.join(
            osp.abspath("."),
            "generalization_ckpt_ppo",
            "ppo_environment_num={},start_seed={}.npz".format(env_num, start_seed)
        )
        # print(trial_path, env_num, start_seed, save_path)
        remove_useless_state(trial_path, save_path)
