import os.path

from haco.algo.haco.haco import HACOTrainer
from haco.utils.human_in_the_loop_env import HumanInTheLoopEnv
from haco.utils.train_utils import initialize_ray


def get_function(exp_path, ckpt_idx):
    ckpt = os.path.join(exp_path, "checkpoint_{}".format(ckpt_idx), "checkpoint-{}".format(ckpt_idx))
    trainer = HACOTrainer(dict(env=HumanInTheLoopEnv))

    trainer.restore(ckpt)

    def _f(obs):
        ret = trainer.compute_actions({"default_policy": obs})
        return ret

    return _f


if __name__ == '__main__':
    # hyperparameters
    CKPT_PATH = "D:\\code\\HACO\\haco\\run_main_exp\\HACO_221223-002810\\HACO_HumanInTheLoopEnv_b0fec_00000_0_seed=0_2022-12-23_00-28-15"
    EPISODE_NUM_PER_CKPT = 2
    CKPT_START = 54
    CKPT_END = 55
    RENDER=False
    env_config = {
        "manual_control": True,
        "use_render": True,
        "controller": "keyboard",
        "window_size": (1600, 1100),
        "cos_similarity": True,
        "map": "CTO",
        "environment_num": 1,
        "start_seed": 15,
    }

    initialize_ray(test_mode=False, local_mode=False, num_gpus=0)


    def make_env(env_cfg=None):
        env_cfg = env_cfg or {}
        env_cfg.update(dict(manual_control=False, use_render=RENDER))
        return HumanInTheLoopEnv(env_cfg)


    from collections import defaultdict

    super_data = defaultdict(list)
    env = make_env(env_config)
    for ckpt_idx in range(CKPT_START, CKPT_END):
        ckpt = ckpt_idx

        compute_actions = get_function(CKPT_PATH, ckpt_idx)

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
            action_to_send = compute_actions(o)["default_policy"]
            o, r, d, info = env.step(action_to_send)
            total_reward += r
            ep_reward += r
            total_cost += info["cost"]
            ep_cost += info["cost"]
            if d or step > horizon:
                if info["arrive_dest"]:
                    success_rate += 1
                    success_flag = True
                epi_num += 1
                super_data[ckpt].append({"reward": ep_reward, "success": success_flag, "cost": ep_cost})

                ep_cost = 0.0
                ep_reward = 0.0
                success_flag = False
                step = 0

                if epi_num >= EPISODE_NUM_PER_CKPT:
                    break
                else:
                    o = env.reset()

        print(
            "CKPT:{} | success_rate:{}, mean_episode_reward:{}, mean_episode_cost:{}".format(ckpt,
                                                                                             success_rate / EPISODE_NUM_PER_CKPT,
                                                                                             total_reward / EPISODE_NUM_PER_CKPT,
                                                                                             total_cost / EPISODE_NUM_PER_CKPT))

        del compute_actions

    env.close()

    import json

    try:
        with open("eval_haco_ret.json", "w") as f:
            json.dump(super_data, f)
    except:
        pass

    print(super_data)
