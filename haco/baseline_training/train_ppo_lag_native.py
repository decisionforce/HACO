import copy

from drivingforce.expert_in_the_loop.common import SaverCallbacks
from drivingforce.expert_in_the_loop.human_in_the_loop_env import HumanInTheLoopEnv
from drivingforce.human_in_the_loop.common import baseline_eval_config
from drivingforce.safety.ppo_lag import PPOLag
from drivingforce.train import train, get_train_parser

evaluation_config = {"env_config": copy.deepcopy(baseline_eval_config)}

if __name__ == '__main__':
    args = get_train_parser().parse_args()

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
        train_batch_size=tune.grid_search([4000, 8000]),
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
        # num_seeds=2,
        num_seeds=10,
        custom_callback=SaverCallbacks,
        # test_mode=True,
        # local_mode=True

        wandb_key_file="~/wandb_api_key_file.txt",
        wandb_project="iclr22",

    )

