from drivingforce.expert_in_the_loop.human_in_the_loop_env import HumanInTheLoopEnv
import pickle
from metadrive.policy.env_input_policy import EnvInputPolicy
from drivingforce.expert_in_the_loop.egpo.sac_pid_saver import SACPIDSaverTrainer
from drivingforce.train.utils import initialize_ray

initialize_ray(test_mode=False)


if __name__ == "__main__":
    env_config = dict(
        use_render=True,
        manual_control=False,
        in_replay=True,
        window_size=(1600, 1080),
        show_fps=False,
        vehicle_config=dict(show_navi_mark=False),
        show_interface_navi_mark=False,
        show_mouse=False,
        show_logo=False,
        camera_dist=7.5
    )
    env = HumanInTheLoopEnv(env_config)

    epi_num = 0

    total_cost = 0
    total_reward = 0
    success_rate = 0
    ep_cost = 0
    ep_reward = 0
    success_flag = False
    horizon = 1000
    step = 0

    ckpt = 169
    for env_index in [102, 106, 109, 112, 114]:
        episode = open("./haco_traj/haco_env_{}_{}.pkl".format(ckpt, env_index), "rb+")
        epi_data = pickle.load(episode)
        episode.close()
        env.config["replay_episode"] = epi_data
        o = env.reset()
        env.main_camera.set_follow_lane(True)
        while True:
            # action_to_send = compute_actions(w, [o], deterministic=False)[0]
            step += 1
            # a = [action_to_send[0], action_to_send[1]] + env.human_intention
            o, r, d, info = env.step([0,0])
            # env.render(text={"env_seed": env_index})
            total_reward += r
            ep_reward += r
            total_cost += info["cost"]
            ep_cost += info["cost"]
            if d or step > horizon or info.get("replay_done", False):
                step=0
                break
