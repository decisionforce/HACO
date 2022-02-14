import os.path as osp
from drivingforce.expert_in_the_loop.egpo.sac_pid_saver import SACPIDSaverTrainer
from drivingforce.human_in_the_loop.prototype.prototype_env import PrototypeEnv
import numpy as np
from drivingforce.train.utils import initialize_ray
import pickle
from core.haco_env import HACOEnv

import numpy as np

# root = ~/.../copo/copo/maour_environment
root = osp.dirname(osp.abspath(__file__))

_checkpoints_buffers = {}


def relu(x):
    return np.clip(x, 0, None)


def sac_policy(weights, obs, deterministic=False):

    obs = np.asarray(obs)
    if obs.ndim == 1:
        obs = np.expand_dims(obs, axis=0)
    assert obs.ndim == 2

    x = np.matmul(obs, weights["default_policy/sequential/action_1/kernel"]) + \
        weights["default_policy/sequential/action_1/bias"]

    x = relu(x)

    x = np.matmul(x, weights["default_policy/sequential/action_2/kernel"]) + \
        weights["default_policy/sequential/action_2/bias"]

    x = relu(x)

    x = np.matmul(x, weights["default_policy/sequential/action_out/kernel"]) + \
        weights["default_policy/sequential/action_out/bias"]

    mean, log_std = np.split(x, 2, axis=1)
    std = np.exp(log_std)
    action = np.random.normal(mean, std) if not deterministic else mean
    squashed = ((np.tanh(action) + 1.0) / 2.0) * 2 - 1
    return squashed


def read_weight(ckpt_path, remove_value_network=True):
    with open(ckpt_path, "rb") as f:
        data = f.read()
    unpickled = pickle.loads(data)
    worker = pickle.loads(unpickled.pop("worker"))
    if "_optimizer_variables" in worker["state"]["default_policy"]:
        worker["state"]["default_policy"].pop("_optimizer_variables")
    weights = worker["state"]["default_policy"]
    if remove_value_network:

        new_weights = {}
        for k, v in weights.items():
            should_use_this_item = True
            for remove_key in ["twin_q", "cost_q", "q_hidden", "q_out", "value", "alpha"]:
                if remove_key in k:
                    should_use_this_item = False
            if should_use_this_item:
                new_weights[k] = v
        return new_weights
    else:
        return weights


class PolicyFunction:
    def __init__(self, ckpt):
        global _checkpoints_buffers
        if ckpt not in _checkpoints_buffers:
            w = read_weight(ckpt)
            _checkpoints_buffers[ckpt] = w
        else:
            w = _checkpoints_buffers[ckpt]
        self.w = w
        self.ckpt = ckpt

    def policy(self, obs, deterministic=False):
        return sac_policy(self.w, obs, deterministic=deterministic)

    def __call__(self, obs, deterministic=False):
        actions = self.policy(obs, deterministic=deterministic)
        # print(actions)
        return actions

    def reset(self):
        pass

class VisionPolicyFunction:
    def __init__(self, ckpt):

        self.trainer = SACPIDSaverTrainer(dict(

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
        ))

        self.trainer.restore(ckpt)

    def __call__(self, obs):
        return self.trainer.compute_actions({"default_policy": obs})

    def __del__(self):
        del self.trainer

    def reset(self):
        pass



