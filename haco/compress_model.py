import pickle

import numpy as np


def compress_model(ckpt_path, path="safe_expert.npz", remove_value_network=False):
    with open(ckpt_path, "rb") as f:
        data = f.read()
    unpickled = pickle.loads(data)
    worker = pickle.loads(unpickled.pop("worker"))
    if "_optimizer_variables" in worker["state"]["default_policy"]:
        worker["state"]["default_policy"].pop("_optimizer_variables")
    pickled_worker = pickle.dumps(worker)
    weights = worker["state"]["default_policy"]
    if remove_value_network:
        weights = {k: v for k, v in weights.items() if "value" not in k}
    np.savez_compressed(path, **weights)
    print("Numpy agent weight is saved at: {}!".format(path))
    
def relu(x):
    return np.clip(x, 0, None)

def conditional_controller(intention, obs, weights, deterministic=True):
    origin_obs = obs
    obs = obs.reshape(1, -1)
    # intention = np.matmul(obs, weights["default_policy/sequential/intention_1/kernel"]) + weights[
    #     "default_policy/sequential/intention_1/bias"]
    # intention = relu(intention)
    # intention = np.matmul(intention, weights["default_policy/sequential/intention_2/kernel"]) + weights[
    #     "default_policy/sequential/intention_2/bias"]
    # intention = relu(intention)
    # intention = np.matmul(intention, weights["default_policy/sequential/intention_out/kernel"]) + weights[
    #     "default_policy/sequential/intention_out/bias"]
    # intention = intention.reshape(-1)
    # index = np.argmax(intention)
    # intention = np.array([0,0,0])
    # intention[index] = 1.0
    # obs = np.concatenate([obs])
    obs = obs.reshape(1, -1)
    x = np.matmul(obs, weights["default_policy/sequential_1/low_level_action_hidden_0/kernel"]) + weights[
        "default_policy/sequential_1/low_level_action_hidden_0/bias"]
    x = relu(x)
    x = np.matmul(x, weights["default_policy/sequential_1/low_level_action_hidden_1/kernel"]) + weights[
        "default_policy/sequential_1/low_level_action_hidden_1/bias"]
    x = relu(x)
    x = np.matmul(x, weights["default_policy/sequential_1/low_level_action_out/kernel"]) + weights[
        "default_policy/sequential_1/low_level_action_out/bias"]

    x = np.split(x, 3, axis=1)
    if intention[0] == 1.0:
        x = x[0]
    elif intention[1] == 1.:
        x = x[1]
    elif intention[2] == 1.:
        x = x[2]
    x = x.reshape(-1)
    mean, log_std = np.split(x, 2)
    std = np.exp(log_std)
    expert_action = np.random.normal(mean, std) if not deterministic else mean
    return expert_action, intention


if __name__ == "__main__":
    ckpt = "checkpoint_146/checkpoint-146"
    compress_model(ckpt, "controller.npz")
