from drivingforce.expert_in_the_loop.egpo.sac_pid_saver import SACPIDSaverTrainer
from drivingforce.human_in_the_loop.prototype.prototype_env import PrototypeEnv
import numpy as np
from drivingforce.train.utils import initialize_ray

initialize_ray(test_mode=False)


def get_function(ckpt):
    trainer = SACPIDSaverTrainer(dict(

        env=PrototypeEnv,

        # ===== Training =====
        takeover_data_discard=False,
        alpha=10.0,
        recent_episode_num=5,
        normalize=True,
        twin_cost_q=True,
        # no_reward=tune.grid_search([True, False]),
        k_i=0.01,
        k_p=5,
        # search > 0
        k_d=0.1,
        # k_i=tune.grid_search([0.01, 0.005, 0.001]),

        # expected max takeover num
        cost_limit=300,
        optimization=dict(actor_learning_rate=1e-4, critic_learning_rate=1e-4, entropy_learning_rate=1e-4),
        prioritized_replay=False,
        horizon=400,
        target_network_update_freq=1,
        timesteps_per_iteration=100,
        metrics_smoothing_episodes=10,
        learning_starts=100,
        clip_actions=False,
        normalize_actions=True,

    ))

    trainer.restore(ckpt)

    def _f(obs):
        ret = trainer.compute_actions({"default_policy": obs})
        return ret

    return _f


if __name__ == '__main__':
    def make_env(env_id=None):
        return PrototypeEnv(dict(manual_control=False, use_render=True, vehicle_config=dict(use_saver=False)))


    env = make_env()

    compute_actions = get_function("../checkpoint_146/checkpoint-146")

    o = env.reset()
    epi_num = 0

    total_cost = 0
    total_reward = 0
    success_rate = 0
    ep_cost = 0
    ep_reward = 0
    success_flag = False
    horizon = 2000
    step = 0
    while True:
        # action_to_send = compute_actions(w, [o], deterministic=False)[0]
        step += 1
        action_to_send = compute_actions(o)
        action_to_send = action_to_send["default_policy"]
        # a = [action_to_send[0], action_to_send[1]] + env.human_intention
        o, r, d, info = env.step(action_to_send)
        env.render(text={"agent_intention":action_to_send[-3:]})
        total_reward += r
        ep_reward += r
        total_cost += info["cost"]
        ep_cost += info["cost"]
        if d or step > horizon:
            if info["arrive_dest"]:
                success_rate += 1
                success_flag = True
            epi_num += 1
            # if epi_num > EPISODE_NUM:
            #     break
            # else:
            o = env.reset()

            # super_data[ckpt].append({"reward": ep_reward, "success": success_flag, "cost": ep_cost})

            ep_cost = 0.0
            ep_reward = 0.0
            success_flag = False
            step = 0
