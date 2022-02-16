import copy
import os
from ray import tune
import ray
from drivingforce.offline_rl.cql.cql import CQLTrainer
from ray.rllib.offline.shuffled_input import ShuffledInput

from drivingforce.expert_in_the_loop.common import ILCallBack
from drivingforce.expert_in_the_loop.human_in_the_loop_env import HumanInTheLoopEnv
from drivingforce.human_in_the_loop.baselines.input_reader import HumanDataInputReader
from drivingforce.human_in_the_loop.common import baseline_eval_config
from drivingforce.train import get_train_parser
from drivingforce.train.train import train

data_set_file_path = expert_value_weights = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                                         "human_traj_100.json")

eval_config = {"env_config": copy.deepcopy(baseline_eval_config)}
eval_config["input"] = "sampler"  # important to use pgdrive online evaluation
eval_config["env_config"]["random_spawn"] = tune.grid_search([True, False])
def get_data_sampler_func(ioctx):
    return ShuffledInput(HumanDataInputReader(data_set_file_path))


if __name__ == '__main__':
    print(data_set_file_path)
    assert ray.__version__ == "1.3.0" or ray.__version__ == "1.2.0", "ray 1.3.0 is required"
    args = get_train_parser().parse_args()

    exp_name = args.exp_name or "CQL"
    stop = {"timesteps_total": 100_0000_00000}

    config = dict(
        # ===== Evaluation =====
        env=HumanInTheLoopEnv,
        env_config=baseline_eval_config,
        input_evaluation=["simulation"],
        evaluation_interval=1,
        evaluation_num_episodes=30,
        evaluation_config=eval_config,
        evaluation_num_workers=2,
        metrics_smoothing_episodes=30,

        # ===== Training =====

        # cql para
        lagrangian=False,  # Automatic temperature (alpha prime) control
        temperature=5,  # alpha prime in paper, 5 is best in metadrive
        min_q_weight=0.2,  # best
        bc_iters=tune.grid_search([5_0000, 10_0000]),  # bc_iters > 20_0000 has no obvious improvement

        # offline setting
        no_done_at_end=True,
        input=get_data_sampler_func,
        optimization=dict(actor_learning_rate=1e-4, critic_learning_rate=1e-4, entropy_learning_rate=1e-4),
        rollout_fragment_length=200,
        prioritized_replay=False,
        horizon=2000,
        target_network_update_freq=1,
        timesteps_per_iteration=1000,
        learning_starts=10000,
        clip_actions=False,
        normalize_actions=True,
        num_cpus_for_driver=0.5,
        # No extra worker used for learning. But this config impact the evaluation workers.
        num_cpus_per_worker=0.1,
        # num_gpus_per_worker=0.1 if args.num_gpus != 0 else 0,
        num_gpus=0.2 if args.num_gpus != 0 else 0,
        framework="torch"
    )

    train(
        CQLTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        # num_seeds=2,
        num_seeds=5,
        custom_callback=ILCallBack,
        # test_mode=True,
        # local_mode=True

        wandb_key_file="~/wandb_api_key_file.txt",
        wandb_project="iclr22",

    )
