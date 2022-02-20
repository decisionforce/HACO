from haco.utils.callback import HACOCallbacks
from drivingforce.expert_in_the_loop.egpo.sac_pid_saver import SACPIDSaverTrainer
from haco.utils.human_in_the_loop_env import HumanInTheLoopEnv
from haco.utils.train import train
from haco.utils.train_utils import get_train_parser
import datetime


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
            "window_size": (1600, 1100),
            "only_takeover_start_cost": True,
            "cos_similarity": True,
        },

        # ===== Training =====
        takeover_data_discard=False,
        recent_episode_num=3,
        normalize=True,
        twin_cost_q=True,
        k_i=0.01,
        k_p=5,
        # search > 0
        k_d=0.1,
        # k_i=tune.grid_search([0.01, 0.005, 0.001]),
        alpha=10,
        no_reward=True,  # need reward
        # search me
        use_td_takeover_mask=False,
        explore=False,

        # expected max takeover num
        cost_limit=-1,
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
        SACPIDSaverTrainer,
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
