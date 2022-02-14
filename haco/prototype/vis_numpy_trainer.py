import numpy as np

from drivingforce.human_in_the_loop.compress_model import conditional_controller
from drivingforce.human_in_the_loop.prototype.prototype_env import PrototypeEnv
from drivingforce.train.utils import initialize_ray

initialize_ray(test_mode=False)


def get_function(npz_ckpt):
    weights = np.load(npz_ckpt)

    def _f(obs, intention):
        return conditional_controller(intention, obs, weights)

    return _f


if __name__ == '__main__':
    def make_env(env_id=None):
        return PrototypeEnv(dict(manual_control=False, use_render=True, vehicle_config=dict(use_saver=False)))


    env = make_env()

    compute_actions = get_function("../controller.npz")

    o = env.reset()
    env.need_external_intention()
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
        action_to_send, agent_intention = compute_actions(obs=o, intention=env.external_intention)
        print(action_to_send)
        action_to_send = np.concatenate([action_to_send, np.array(env.human_intention)])
        o, r, d, info = env.step(action_to_send)
        env.render(text={"agent_intention": env.external_intention})
        total_reward += r
        ep_reward += r
        total_cost += info["cost"]
        ep_cost += info["cost"]
        if False:
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
