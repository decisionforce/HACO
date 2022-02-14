from __future__ import print_function

import gzip
import json
import os
import pickle
import time

import numpy as np
import torch


def save_results(episode_rewards, results_dir="./results", result_file_name="training_result"):
    # save results
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # save statistics in a dictionary and write them into a .json file
    results = dict()
    results["number_episodes"] = len(episode_rewards)
    results["episode_rewards"] = episode_rewards

    results["mean_all_episodes"] = np.array(episode_rewards).mean()
    results["std_all_episodes"] = np.array(episode_rewards).std()

    fname = os.path.join(results_dir, result_file_name)
    fh = open(fname, "w")
    json.dump(results, fh)
    print('... finished')


def store_data(data, datasets_dir="./data", num_epoch=0):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'data_dagger_{}.pkl.gzip'.format(num_epoch))
    f = gzip.open(data_file, 'wb')
    pickle.dump(data, f)


def read_data(datasets_dir="./data", path='data.pkl.gzip', frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    data_file = os.path.join(datasets_dir, path)

    f = gzip.open(data_file, 'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = data["state"]
    y = data["action"]
    return X, y


def shuffle_data(X_train, y_train):
    # shuffle
    perm = np.arange(len(X_train))
    np.random.shuffle(perm)
    X_train = X_train[perm]
    y_train = y_train[perm]
    return X_train, y_train


def train_model(model, X_train, y_train, path, num_epochs=50, learning_rate=1e-3, lambda_l2=1e-5,
                batch_size=32, shuffle=True, exp_log=None, log_interval=10):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    criterion = torch.nn.MSELoss()
    optimizer = [torch.optim.SGD(model.pis[i].parameters(), lr=learning_rate, weight_decay=lambda_l2) for i in
                 range(model.num_nets)]  # built-in L2
    train_start = time.time()
    X_train_torch = torch.from_numpy(X_train).to(model.device).float()
    y_train_torch = torch.from_numpy(y_train).to(model.device).float()
    total_loss = []
    sgd_num = len(X_train_torch) / batch_size
    for t in range(num_epochs):
        if shuffle:
            X_train, y_train = shuffle_data(X_train, y_train)
        epoch_loss = 0
        for i in range(0, len(X_train_torch), batch_size):
            curr_X = X_train_torch[i:i + batch_size]
            curr_Y = y_train_torch[i:i + batch_size]
            for i in range(model.num_nets):
                preds = model.pis[i](curr_X)
                loss = criterion(preds, curr_Y)
                total_loss.append(loss.item())
                epoch_loss += loss.item()
                optimizer[i].zero_grad()
                loss.backward()
                optimizer[i].step()
        if exp_log is not None and t % log_interval == 0:
            current_time = time.time()
            exp_log.scalar(is_train=True,
                           data_set_size=len(X_train),
                           epoch_loss=epoch_loss / sgd_num / model.num_nets,
                           epoch_training_time=(current_time - train_start) / log_interval,
                           ensemble_variance=model.variance(X_train))
            train_start = current_time
    exp_log.scalar(is_train=True,
                   last_epoch_loss=epoch_loss / sgd_num / model.num_nets,
                   total_sgd_epoch_num=num_epochs)
    model.save(path)


def evaluation(env, model, evaluation_episode_num=30, exp_log=None):
    device = model.device
    with torch.no_grad():
        print("... evaluation")
        episode_reward = 0
        episode_cost = 0
        success_num = 0
        episode_num = 0
        velocity = []
        episode_overtake = []
        state = env.reset()
        while episode_num < evaluation_episode_num:
            prediction = model(torch.tensor(state).to(device).float())
            next_state, r, done, info = env.step(prediction.detach().cpu().numpy().flatten())
            state = next_state
            episode_reward += r
            episode_cost += info["native_cost"]
            velocity.append(info["velocity"])
            if done:
                episode_overtake.append(info["overtake_vehicle_num"])
                episode_num += 1
                if info["arrive_dest"]:
                    success_num += 1
                env.reset()
        res = dict(
            mean_episode_reward=episode_reward / episode_num,
            mean_episode_cost=episode_cost / episode_num,
            mean_success_rate=success_num / episode_num,
            mean_velocity=np.mean(velocity),
            mean_episode_overtake_num=np.mean(episode_overtake)
        )
        if exp_log is not None:
            exp_log.scalar(is_train=False, **res)
        return res


def load_human_data(path, data_usage=5000):
    """
   This method reads the states and actions recorded by human expert in the form of episode
   """
    with open(path, "r") as f:
        episode_data = json.load(f)["data"]
    np.random.shuffle(episode_data)
    assert data_usage < len(episode_data), "Data is not enough"
    data = {"state": [],
            "action": [],
            "next_state": [],
            "reward": [],
            "terminal": []}
    for cnt, step_data in enumerate(episode_data):
        if cnt >= data_usage:
            break
        data["state"].append(step_data["obs"])
        data["next_state"].append(step_data["new_obs"])
        data["action"].append(step_data["actions"])
        data["terminal"].append(step_data["dones"])
    # get images as features and actions as targets
    return data
