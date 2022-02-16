import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import datetime
from typing import Dict

import numpy as np
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from core.haco_env import HACOEnv

from drivingforce.expert_in_the_loop.egpo.sac_pid_saver import SACPIDSaverTrainer
from haco.utils.train import train
from haco.utils.train_utils import get_train_parser

from ray.rllib.agents.callbacks import DefaultCallbacks


class CARLACallBack(DefaultCallbacks):
    def on_episode_start(
            self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
            env_index: int, **kwargs
    ):
        episode.user_data["velocity"] = []
        episode.user_data["steering"] = []
        episode.user_data["step_reward"] = []
        episode.user_data["acceleration"] = []
        episode.user_data["takeover"] = 0
        episode.user_data["raw_episode_reward"] = 0
        episode.user_data["episode_crash_rate"] = 0
        episode.user_data["episode_out_of_road_rate"] = 0
        episode.user_data["total_takeover_cost"] = 0
        episode.user_data["total_native_cost"] = 0
        episode.user_data["cost"] = 0

    def on_episode_step(
            self, *, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, env_index: int, **kwargs
    ):
        info = episode.last_info_for()
        if info is not None:
            episode.user_data["velocity"].append(info["velocity"])
            episode.user_data["steering"].append(info["steering"])
            episode.user_data["acceleration"].append(info["acceleration"])
            episode.user_data["step_reward"].append(info["step_reward"])
            episode.user_data["takeover"] += 1 if info["takeover"] else 0
            episode.user_data["raw_episode_reward"] += info["step_reward"]
            episode.user_data["episode_crash_rate"] += 1 if info["crash"] else 0
            episode.user_data["episode_out_of_road_rate"] += 1 if info["out_of_road"] else 0
            # episode.user_data["high_speed_rate"] += 1 if info["high_speed"] else 0
            episode.user_data["total_takeover_cost"] += info["takeover_cost"]
            episode.user_data["total_native_cost"] += info["native_cost"]
            episode.user_data["cost"] += info["cost"] if "cost" in info else info["native_cost"]


    def on_episode_end(
            self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
            **kwargs) -> None:
        arrive_dest = episode.last_info_for()["arrive_dest"]
        crash = episode.last_info_for()["crash"]
        out_of_road = episode.last_info_for()["out_of_road"]
        max_step_rate = not (arrive_dest or crash or out_of_road)
        episode.custom_metrics["success_rate"] = float(arrive_dest)
        episode.custom_metrics["crash_rate"] = float(crash)
        episode.custom_metrics["out_of_road_rate"] = float(out_of_road)
        episode.custom_metrics["max_step_rate"] = float(max_step_rate)
        episode.custom_metrics["velocity_max"] = float(np.max(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_mean"] = float(np.mean(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_min"] = float(np.min(episode.user_data["velocity"]))
        episode.custom_metrics["steering_max"] = float(np.max(episode.user_data["steering"]))
        episode.custom_metrics["steering_mean"] = float(np.mean(episode.user_data["steering"]))
        episode.custom_metrics["steering_min"] = float(np.min(episode.user_data["steering"]))
        episode.custom_metrics["acceleration_min"] = float(np.min(episode.user_data["acceleration"]))
        episode.custom_metrics["acceleration_mean"] = float(np.mean(episode.user_data["acceleration"]))
        episode.custom_metrics["acceleration_max"] = float(np.max(episode.user_data["acceleration"]))
        episode.custom_metrics["step_reward_max"] = float(np.max(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_mean"] = float(np.mean(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_min"] = float(np.min(episode.user_data["step_reward"]))
        episode.custom_metrics["takeover_rate"] = float(episode.user_data["takeover"] / episode.length)
        episode.custom_metrics["takeover_count"] = float(episode.user_data["takeover"])
        episode.custom_metrics["raw_episode_reward"] = float(episode.user_data["raw_episode_reward"])
        episode.custom_metrics["episode_crash_num"] = float(episode.user_data["episode_crash_rate"])
        episode.custom_metrics["episode_out_of_road_num"] = float(episode.user_data["episode_out_of_road_rate"])

        episode.custom_metrics["total_takeover_cost"] = float(episode.user_data["total_takeover_cost"])
        episode.custom_metrics["total_native_cost"] = float(episode.user_data["total_native_cost"])

        episode.custom_metrics["cost"] = float(episode.user_data["cost"])


    def on_train_result(self, *, trainer, result: dict, **kwargs):
        result["success"] = np.nan
        result["crash"] = np.nan
        result["out"] = np.nan
        result["max_step"] = np.nan
        result["cost"] = np.nan
        result["length"] = result["episode_len_mean"]
        result["takeover"] = np.nan
        if "success_rate_mean" in result["custom_metrics"]:
            result["success"] = result["custom_metrics"]["success_rate_mean"]
            result["crash"] = result["custom_metrics"]["crash_rate_mean"]
            result["out"] = result["custom_metrics"]["out_of_road_rate_mean"]
            result["max_step"] = result["custom_metrics"]["max_step_rate_mean"]
            result["native_cost"] = result["custom_metrics"]["total_native_cost_mean"]
        if "cost_mean" in result["custom_metrics"]:
            result["cost"] = result["custom_metrics"]["cost_mean"]
        if "takeover_count_mean" in result["custom_metrics"]:
            result["takeover"] = result['custom_metrics']["takeover_count_mean"]

    # turn on overtake stata only in evaluation

def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")




if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    args = get_train_parser().parse_args()

    exp_name = args.exp_name or "EGPO_CARLA_{}".format(get_time_str())
    stop = {"timesteps_total": 8_0000}

    config = dict(
        env=HACOEnv,
        env_config={},

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
        image_obs=True,

        # expected max takeover num
        cost_limit=-1,
        explore=False,
        optimization=dict(actor_learning_rate=1e-5, critic_learning_rate=1e-4, entropy_learning_rate=1e-4),
        prioritized_replay=True,
        horizon=1500,
        target_network_update_freq=1,
        timesteps_per_iteration=100,
        metrics_smoothing_episodes=10,
        learning_starts=100,
        clip_actions=False,
        train_batch_size=128,

        normalize_actions=True,
        num_cpus_for_driver=0.5,
        # No extra worker used for learning. But this config impact the evaluation workers.
        num_cpus_per_worker=0.1,
        # num_gpus_per_worker=0.1 if args.num_gpus != 0 else 0,
        num_gpus=0.2,
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
        custom_callback=CARLACallBack,
        # test_mode=True,
        # local_mode=True
    )
