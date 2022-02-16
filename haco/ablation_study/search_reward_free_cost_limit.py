from ray import tune
from drivingforce.expert_in_the_loop.egpo.sac_pid_saver import SACPIDSaverTrainer
from drivingforce.expert_in_the_loop.expert_guided_env import ExpertGuidedEnv
from haco.utils.callback import HACOCallbacks, evaluation_config
from drivingforce.train import train, get_train_parser
import os

if __name__ == '__main__':
    args = get_train_parser().parse_args()
    expert_value_weights = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "expert.npz")

    exp_name = args.exp_name or "EGPO"
    stop = {"timesteps_total": 20_0000}

    config = dict(
        env=ExpertGuidedEnv,
        env_config=dict(
            vehicle_config=dict(
                use_saver=True,
                free_level=0.99,
            ),
            expert_value_weights=expert_value_weights,
            cost_to_reward=True,
            # rule_takeover=True,
            safe_rl_env=True),

        # ===== Evaluation =====
        evaluation_interval=1,
        evaluation_num_episodes=30,
        evaluation_config=evaluation_config,
        evaluation_num_workers=2,
        metrics_smoothing_episodes=30,

        # ===== Training =====
        takeover_data_discard=False,
        alpha=3.0,
        recent_episode_num=5,
        normalize=True,
        twin_cost_q=True,
        no_reward=True,
        k_i=0.01,
        k_p=5,
        # search > 0
        k_d=0.1,
        # k_i=tune.grid_search([0.01, 0.005, 0.001]),

        # expected max takeover num
        cost_limit=tune.grid_search([0, 10, 20, 50]),
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
        SACPIDSaverTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        # num_seeds=2,
        num_seeds=4,
        custom_callback=SaverCallbacks,
        # test_mode=True,
        # local_mode=True
    )
