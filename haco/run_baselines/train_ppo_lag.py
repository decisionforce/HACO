import copy

import ray
from haco.algo.ppo_lag.ppo_lag import PPOLag
from haco.utils.callback import HACOCallbacks
from haco.utils.config import baseline_eval_config
from haco.utils.human_in_the_loop_env import HumanInTheLoopEnv
from haco.utils.train import train
from haco.utils.train_utils import get_train_parser

evaluation_config = {"env_config": copy.deepcopy(baseline_eval_config)}

if __name__ == '__main__':
    args = get_train_parser().parse_args()
    assert ray.__version__ == "1.0.0", "ray 1.0.0 is required for PPO-Lag"
    exp_name = args.exp_name or "PPO_LAG"
    stop = {"timesteps_total": 1000_0000}

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
        cost_limit=1,
        horizon=1500,
        num_sgd_iter=20,
        lr=5e-5,
        grad_clip=10.0,
        rollout_fragment_length=200,
        sgd_minibatch_size=100,
        train_batch_size=4000,
        num_gpus=0.2 if args.num_gpus != 0 else 0,
        num_cpus_per_worker=0.1,
        num_cpus_for_driver=0.5,
        num_workers=8,
        clip_actions=False
    )

    train(
        PPOLag,
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        num_seeds=5,
        custom_callback=HACOCallbacks,
        # test_mode=True,
        # local_mode=True
    )
