from ray.rllib.agents.sac.sac import SACTrainer
from ray import tune
from haco.utils.human_in_the_loop_env import HumanInTheLoopEnv
from haco.utils.callback import HACOCallbacks, evaluation_config
from drivingforce.train import train, get_train_parser

if __name__ == '__main__':
    args = get_train_parser().parse_args()

    exp_name =  args.exp_name or "SAC_baseline"
    stop = {"timesteps_total": 100_0000}

    config = dict(
        env=HumanInTheLoopEnv,
        env_config=dict(
            crash_vehicle_penalty=1.,
            crash_object_penalty=0.5,
            out_of_road_penalty=1.,
            safe_rl_env=True,
            manual_control=False,
            use_render=False
        ),

        # ===== Evaluation =====
        evaluation_interval=1,
        evaluation_num_episodes=30,
        evaluation_config=dict(env_config=dict(
            start_seed=300,
            environment_num=50,
        )),
        evaluation_num_workers=2,
        metrics_smoothing_episodes=100,

        # ===== Training =====
        optimization=dict(actor_learning_rate=1e-4, critic_learning_rate=1e-4, entropy_learning_rate=1e-4),
        prioritized_replay=False,
        horizon=2000,
        target_network_update_freq=1,
        timesteps_per_iteration=1000,
        learning_starts=10000,
        clip_actions=False,
        normalize_actions=True,
        num_cpus_for_driver=1,
        # No extra worker used for learning. But this config impact the evaluation workers.
        num_cpus_per_worker=0.5,
        # num_gpus_per_worker=0.1 if args.num_gpus != 0 else 0,
        num_gpus=0.5 if args.num_gpus != 0 else 0,
    )

    train(
        SACTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        # num_seeds=1,
        num_seeds=5,
        custom_callback=SaverCallbacks,
        # test_mode=True,
        # local_mode=True
        wandb_key_file="~/wandb_api_key_file.txt",
        wandb_project="iclr22",
    )
