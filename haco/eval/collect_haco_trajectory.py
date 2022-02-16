from haco.utils.human_in_the_loop_env import HumanInTheLoopEnv
from metadrive.policy.env_input_policy import EnvInputPolicy
from drivingforce.expert_in_the_loop.egpo.sac_pid_saver import SACPIDSaverTrainer
from drivingforce.safety.ppo_lag import PPOLag
from drivingforce.train.utils import initialize_ray
from ray.rllib.agents.sac.sac import SACTrainer

initialize_ray(test_mode=False)


def get_function(ckpt):
    trainer = SACTrainer(dict(

        env=HumanInTheLoopEnv,

        # # ===== Training =====
        # takeover_data_discard=False,
        # alpha=10.0,
        # recent_episode_num=5,
        # normalize=True,
        # twin_cost_q=True,
        # # no_reward=tune.grid_search([True, False]),
        # k_i=0.01,
        # k_p=5,
        # # search > 0
        # k_d=0.1,
        # # k_i=tune.grid_search([0.01, 0.005, 0.001]),
        #
        # # expected max takeover num
        # cost_limit=300,
        # optimization=dict(actor_learning_rate=1e-4, critic_learning_rate=1e-4, entropy_learning_rate=1e-4),
        # prioritized_replay=False,
        # horizon=400,
        # target_network_update_freq=1,
        # timesteps_per_iteration=100,
        # metrics_smoothing_episodes=10,
        # learning_starts=100,
        # clip_actions=False,
        # normalize_actions=True,
        num_cpus_for_driver=0.5,
        # No extra worker used for learning. But this config impact the evaluation workers.
        num_cpus_per_worker=0.1,
        num_gpus=0,
        framework="torch"

    ))

    trainer.restore(ckpt)

    def _f(obs):
        ret = trainer.compute_actions({"default_policy": obs})
        return ret

    return _f


if __name__ == "__main__":
    algo = "cql"
    env_config = dict(
        use_render=False,
        manual_control=False,
        agent_policy=EnvInputPolicy,
        record_episode=True,
    )
    env = HumanInTheLoopEnv(env_config)
    compute_actions = get_function("./checkpoint_140/checkpoint-140")
    o = env.reset(force_seed=102)
    epi_num = 0

    total_cost = 0
    total_reward = 0
    success_rate = 0
    ep_cost = 0
    ep_reward = 0
    success_flag = False
    horizon = 2000
    step = 0
    for env_index in [106, 109, 112, 114, 119, 123, 133, 137, 149]:
        while True:
            # action_to_send = compute_actions(w, [o], deterministic=False)[0]
            step += 1
            action_to_send = compute_actions(o)
            action_to_send = action_to_send["default_policy"]
            # a = [action_to_send[0], action_to_send[1]] + env.human_intention
            o, r, d, info = env.step(action_to_send)
            # env.render(text={"env_seed": env.current_seed})
            total_reward += r
            ep_reward += r
            total_cost += info["cost"]
            ep_cost += info["cost"]
            if d or step > horizon:
                break

            # if info["arrive_dest"] and ep_cost == 0:
            #     success_rate += 1
            #     success_flag = True
            #     break
        env.engine.dump_episode("./{}_traj/{}_{}.pkl".format(algo,algo,env.current_seed))
        epi_num += 1
        o = env.reset(force_seed=env_index)
        ep_cost = 0.0
        ep_reward = 0.0
        success_flag = False
        step = 0
