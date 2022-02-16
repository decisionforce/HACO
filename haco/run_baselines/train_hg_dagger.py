from __future__ import print_function

import os

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from haco.utils.human_in_the_loop_env import HumanInTheLoopEnv
from algo.HG_Dagger.exp_saver import Experiment
from algo.HG_Dagger import Ensemble
from haco.utils.config import baseline_eval_config, baseline_train_config

# requirement:
# conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=9.2 -c pytorch
# pip install loguru imageio easydict tensorboardX pyyaml stable_baselines3 pickle5 evdev

# hyperpara
BC_WARMUP_DATA_USAGE = 30000  # use human data to do warm up
NUM_ITS = 5
STEP_PER_ITER = 5000
learning_rate = 5e-4
batch_size = 256

need_eval = False  # we do not perform online evaluation. Instead, we evaluate by saved model
evaluation_episode_num = 30
num_sgd_epoch = 1000  # sgd epoch on data set
device = "cuda"

# training env_config/test env config
training_config = baseline_train_config
training_config["use_render"] = True
training_config["manual_control"] = True
eval_config = baseline_eval_config

if __name__ == "__main__":
    tm = time.localtime(time.time())
    tm_stamp = "%s-%s-%s-%s-%s-%s" % (tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)
    log_dir = os.path.join(
        "hg_dagger_lr_{}_bs_{}_sgd_iter_{}_iter_batch_{}".format(learning_rate, batch_size, num_sgd_epoch,
                                                                 STEP_PER_ITER), tm_stamp)
    exp_log = Experiment()
    exp_log.init(log_dir=log_dir)
    model_save_path = os.path.join(log_dir, "hg_dagger_models")
    os.mkdir(model_save_path)

    training_env = SubprocVecEnv(
        [lambda: HumanInTheLoopEnv(training_config)])  # seperate with eval env to avoid metadrive collapse
    eval_env = HumanInTheLoopEnv(eval_config)
    obs_shape = eval_env.observation_space.shape
    action_shape = eval_env.action_space.shape

    # fill buffer with expert data
    samples = load_human_data("../human_traj_100_new.json", data_usage=BC_WARMUP_DATA_USAGE)

    # train first epoch
    agent = Ensemble(obs_shape, action_shape, device=device).to(device).float()
    X_train, y_train = samples["state"], samples["action"]
    train_model(agent, X_train, y_train,
                os.path.join(model_save_path, "model_{}.pth".format(0)),
                num_epochs=num_sgd_epoch,
                batch_size=batch_size,
                learning_rate=learning_rate,
                exp_log=exp_log)
    if need_eval:
        evaluation(eval_env, agent, evaluation_episode_num=evaluation_episode_num, exp_log=exp_log)
    exp_log.end_iteration(0)

    # count
    for iteration in range(1, NUM_ITS):
        steps = 0
        episode_reward = 0
        success_num = 0
        episode_cost = 0
        done_num = 0
        state = training_env.reset()[0]
        # for user friendly :)
        training_env.env_method("stop")
        print("Finish training iteration:{}, Press S to Start new iteration".format(iteration - 1))
        sample_start = time.time()

        while True:

            next_state, r, done, info = training_env.step(np.array([agent.act(torch.tensor(state, device=device))]))
            next_state = next_state[0]
            r = r[0]
            done = done[0]
            info = info[0]
            action = info["raw_action"]
            takeover = info["takeover"]

            episode_reward += r
            episode_cost += info["native_cost"]
            if takeover:
                # for hg dagger aggregate data only when takeover occurs
                samples["state"].append(state)
                samples["action"].append(np.array(action))
                samples["next_state"].append(next_state)
                samples["reward"].append(r)
                samples["terminal"].append(done)

            state = next_state
            steps += 1

            # train after BATCH_PER_ITER steps
            if done:
                if info["arrive_dest"]:
                    success_num += 1
                done_num += 1
                if steps > STEP_PER_ITER:
                    exp_log.scalar(is_train=True, mean_episode_reward=episode_reward / done_num,
                                   mean_episode_cost=episode_cost / done_num,
                                   success_rate=success_num / done_num,
                                   mean_step_reward=episode_reward / steps,
                                   sample_time=time.time() - sample_start,
                                   buffer_size=len(samples))

                    X_train, y_train = samples["state"], samples["action"]
                    # Create new model
                    agent = Ensemble(obs_shape, action_shape, device=device).to(device).float()
                    train_model(agent, X_train, y_train,
                                os.path.join(model_save_path, "model_{}.pth".format(iteration)),
                                num_epochs=num_sgd_epoch,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                exp_log=exp_log)
                    if need_eval:
                        evaluation(eval_env, agent, evaluation_episode_num=evaluation_episode_num, exp_log=exp_log)
                    break
        exp_log.end_iteration(iteration)
    training_env.close()
    eval_env.close()
