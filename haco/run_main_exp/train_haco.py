import datetime

from haco.algo.haco.haco import HACOTrainer
from haco.utils.callback import HACOCallbacks
from haco.utils.human_in_the_loop_env import HumanInTheLoopEnv
from haco.utils.train import train
from haco.utils.train_utils import get_train_parser


def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


if __name__ == '__main__':
    args = get_train_parser().parse_args()

    exp_name = args.exp_name or "HACO_{}".format(get_time_str())
    stop = {"timesteps_total": 8_0000}

    config = dict(
        env=HumanInTheLoopEnv,
        env_config={
            "manual_control": True,
            "use_render": True,
            # "controller": "keyboard",  # use keyboard or not
            "window_size": (1600, 1100),
            "cos_similarity": True,
        },

        # ===== Training =====
        takeover_data_discard=False,
        twin_cost_q=True,
        alpha=10,
        no_reward=True,  # need reward
        explore=True,
        optimization=dict(actor_learning_rate=1e-4, critic_learning_rate=1e-4, entropy_learning_rate=1e-4),
        prioritized_replay=False,
        horizon=1000,
        target_network_update_freq=1,
        timesteps_per_iteration=100,
        metrics_smoothing_episodes=10,
        learning_starts=100,
        clip_actions=False,
        train_batch_size=1024,

        normalize_actions=True,
        num_cpus_for_driver=0.5,
        # No extra worker used for learning. But this config impact the evaluation workers.
        num_cpus_per_worker=0.1,
        # num_gpus_per_worker=0.1 if args.num_gpus != 0 else 0,
        num_gpus=0.2 if args.num_gpus != 0 else 0,
    )

    train(
        HACOTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=None,
        checkpoint_freq=1,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        # num_seeds=2,
        num_seeds=1,
        custom_callback=HACOCallbacks,
        # test_mode=True,
        # local_mode=True
    )
