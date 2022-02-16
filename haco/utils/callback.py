from typing import Dict

import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy


class DrivingCallbacks(DefaultCallbacks):
    def on_episode_start(
            self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
            env_index: int, **kwargs
    ):
        episode.user_data["velocity"] = []
        episode.user_data["steering"] = []
        episode.user_data["step_reward"] = []
        episode.user_data["acceleration"] = []
        episode.user_data["cost"] = []

    def on_episode_step(
            self, *, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, env_index: int, **kwargs
    ):
        info = episode.last_info_for()
        if info is not None:
            episode.user_data["velocity"].append(info["velocity"])
            episode.user_data["steering"].append(info["steering"])
            episode.user_data["step_reward"].append(info["step_reward"])
            episode.user_data["acceleration"].append(info["acceleration"])
            episode.user_data["cost"].append(info["cost"])

    def on_episode_end(
            self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
            **kwargs
    ):
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
        episode.custom_metrics["cost"] = float(sum(episode.user_data["cost"]))

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        result["success"] = np.nan
        result["crash"] = np.nan
        result["out"] = np.nan
        result["max_step"] = np.nan
        result["length"] = result["episode_len_mean"]
        result["cost"] = np.nan
        if "custom_metrics" not in result:
            return

        if "success_rate_mean" in result["custom_metrics"]:
            result["success"] = result["custom_metrics"]["success_rate_mean"]
            result["crash"] = result["custom_metrics"]["crash_rate_mean"]
            result["out"] = result["custom_metrics"]["out_of_road_rate_mean"]
            result["max_step"] = result["custom_metrics"]["max_step_rate_mean"]
        if "cost_mean" in result["custom_metrics"]:
            result["cost"] = result["custom_metrics"]["cost_mean"]


class HACOCallbacks(DrivingCallbacks):
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
        episode.user_data["high_speed_rate"] = 0
        episode.user_data["total_takeover_cost"] = 0
        episode.user_data["total_native_cost"] = 0
        episode.user_data["cost"] = 0
        episode.user_data["episode_crash_vehicle"] = 0
        episode.user_data["episode_crash_object"] = 0

    def on_episode_step(
            self, *, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, env_index: int, **kwargs
    ):
        info = episode.last_info_for()
        if info is not None:
            episode.user_data["velocity"].append(info["velocity"])
            episode.user_data["steering"].append(info["steering"])
            episode.user_data["step_reward"].append(info["step_reward"])
            episode.user_data["acceleration"].append(info["acceleration"])
            episode.user_data["takeover"] += 1 if info["takeover"] else 0
            episode.user_data["raw_episode_reward"] += info["step_reward"]
            episode.user_data["episode_crash_rate"] += 1 if info["crash"] else 0
            episode.user_data["episode_out_of_road_rate"] += 1 if info["out_of_road"] else 0
            # episode.user_data["high_speed_rate"] += 1 if info["high_speed"] else 0
            episode.user_data["total_takeover_cost"] += info["takeover_cost"]
            episode.user_data["total_native_cost"] += info["native_cost"]
            episode.user_data["cost"] += info["cost"] if "cost" in info else info["native_cost"]

            episode.user_data["episode_crash_vehicle"] += 1 if info["crash_vehicle"] else 0
            episode.user_data["episode_crash_object"] += 1 if info["crash_object"] else 0

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
        episode.custom_metrics["high_speed_rate"] = float(episode.user_data["high_speed_rate"] / episode.length)

        episode.custom_metrics["total_takeover_cost"] = float(episode.user_data["total_takeover_cost"])
        episode.custom_metrics["total_native_cost"] = float(episode.user_data["total_native_cost"])

        episode.custom_metrics["cost"] = float(episode.user_data["cost"])
        episode.custom_metrics["overtake_num"] = int(episode.last_info_for()["overtake_vehicle_num"])

        episode.custom_metrics["episode_crash_vehicle_num"] = float(episode.user_data["episode_crash_vehicle"])
        episode.custom_metrics["episode_crash_object_num"] = float(episode.user_data["episode_crash_object"])

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
evaluation_config = dict(env_config=dict(
    vehicle_config=dict(use_saver=False, overtake_stat=False),
    safe_rl_env=True,
    start_seed=500,
    environment_num=50,
    horizon=1000,
))


class ILCallBack(HACOCallbacks):
    def on_train_result(self, *, trainer, result: dict, **kwargs):
        result["success"] = np.nan
        result["crash"] = np.nan
        result["out"] = np.nan
        result["max_step"] = np.nan
        result["cost"] = np.nan
        result["length"] = np.nan
        result["takeover"] = np.nan
        if "evaluation" in result:
            eval = result["evaluation"]
            if "success_rate_mean" in eval["custom_metrics"]:
                result["success"] = eval["custom_metrics"]["success_rate_mean"]
                result["crash"] = eval["custom_metrics"]["crash_rate_mean"]
                result["out"] = eval["custom_metrics"]["out_of_road_rate_mean"]
                result["max_step"] = eval["custom_metrics"]["max_step_rate_mean"]
                result["native_cost"] = eval["custom_metrics"]["total_native_cost_mean"]
            if "cost_mean" in eval["custom_metrics"]:
                result["cost"] = eval["custom_metrics"]["cost_mean"]
            if "takeover_count_mean" in eval["custom_metrics"]:
                result["takeover"] = eval['custom_metrics']["takeover_count_mean"]
            if "episode_reward_mean" in eval:
                result["episode_reward"] = eval["episode_reward_mean"]
                result["episode_reward_mean"] = eval["episode_reward_mean"]
                result["reward"] = eval["episode_reward_mean"]
                result["length"] = eval["episode_len_mean"]