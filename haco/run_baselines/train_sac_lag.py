import copy

import ray
from haco.algo.sac_lag.sac_lag import SACLagTrainer
from haco.utils.callback import HACOCallbacks
from haco.utils.config import baseline_eval_config
from haco.utils.human_in_the_loop_env import HumanInTheLoopEnv
from haco.utils.train import train
from haco.utils.train_utils import get_train_parser

evaluation_config = {"env_config": copy.deepcopy(baseline_eval_config)}

if __name__ == '__main__':
    args = get_train_parser().parse_args()
    assert ray.__version__ == "1.0.0", "ray 1.0.0 is required for SAC-Lag"
    exp_name = args.exp_name or "SAC_Lag"
    stop = int(100_0000)

    config = dict(
        env=HumanInTheLoopEnv,
        env_config=dict(
            main_exp=False
        ),

        # ===== Evaluation =====
        evaluation_interval=1,
        evaluation_num_episodes=30,
        evaluation_config=evaluation_config,
        evaluation_num_workers=2,
        metrics_smoothing_episodes=30,

        # ===== Training =====
        # Best:
        # recent episode num: 3
        recent_episode_num=5,
        normalize=True,
        only_evaluate_cost=False,
        twin_cost_q=True,
        k_i=0.01,
        k_p=5,
        k_d=0.1,
        cost_limit=1,
        optimization=dict(actor_learning_rate=1e-4, critic_learning_rate=1e-4, entropy_learning_rate=1e-4),
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
    )

    train(
        SACLagTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        # num_seeds=args.num_seeds,
        num_seeds=5,
        custom_callback=HACOCallbacks,
        # test_mode=True,
        # local_mode=True
    )
