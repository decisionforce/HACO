import pickle
from metadrive.component.road.road import Road
from metadrive.component.vehicle_module.navigation import Navigation

from panda3d.core import NodePath

import numpy as np
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.static_object.traffic_object import TrafficBarrier
from metadrive.component.blocks.first_block import FirstPGBlock
from panda3d.core import BitMask32

from haco.utils.human_in_the_loop_env import HumanInTheLoopEnv

FirstPGBlock.ENTRANCE_LENGTH = 10
# Navigation.FORCE_CALCULATE = True

def relu(x):
    return np.clip(x, 0, None)


def load_model(ckpt_path):
    with open(ckpt_path, "rb") as f:
        data = f.read()
    unpickled = pickle.loads(data)
    worker = pickle.loads(unpickled.pop("worker"))
    weights = worker["state"]["default_policy"]
    return weights


def controller(obs, weights, deterministic=True):
    obs = obs.reshape(1, -1)
    x = np.matmul(obs, weights["default_policy/sequential/action_1/kernel"]) + weights[
        "default_policy/sequential/action_1/bias"]
    x = relu(x)
    x = np.matmul(x, weights["default_policy/sequential/action_2/kernel"]) + weights[
        "default_policy/sequential/action_2/bias"]
    x = relu(x)
    x = np.matmul(x, weights["default_policy/sequential/action_out/kernel"]) + weights[
        "default_policy/sequential/action_out/bias"]

    x = x.reshape(-1)
    mean, log_std = np.split(x, 2)
    std = np.exp(log_std)
    expert_action = np.random.normal(mean, std) if not deterministic else mean
    return expert_action


def compute_value(obs, action, weights):
    x = np.concatenate([obs, action], axis=-1)
    x = x.reshape(1, -1)
    x = np.matmul(x, weights["default_policy/sequential_1/q_hidden_0/kernel"]) + weights[
        "default_policy/sequential_1/q_hidden_0/bias"]
    x = relu(x)
    x = np.matmul(x, weights["default_policy/sequential_1/q_hidden_1/kernel"]) + weights[
        "default_policy/sequential_1/q_hidden_1/bias"]
    x = relu(x)
    x = np.matmul(x, weights["default_policy/sequential_1/q_out/kernel"]) + weights[
        "default_policy/sequential_1/q_out/bias"]
    return x


def compute_cost_value(obs, action, weights):
    x = np.concatenate([obs, action], axis=-1)
    x = x.reshape(1, -1)
    x = np.matmul(x, weights["default_policy/sequential_2/cost_q_hidden_0/kernel"]) + weights[
        "default_policy/sequential_2/cost_q_hidden_0/bias"]
    x = relu(x)
    x = np.matmul(x, weights["default_policy/sequential_2/cost_q_hidden_1/kernel"]) + weights[
        "default_policy/sequential_2/cost_q_hidden_1/bias"]
    x = relu(x)
    x = np.matmul(x, weights["default_policy/sequential_2/cost_q_out/kernel"]) + weights[
        "default_policy/sequential_2/cost_q_out/bias"]
    return x


def set_state(vehicle: BaseVehicle, position, heading_theta=0):
    vehicle.set_position(position)
    vehicle.set_velocity([np.cos(heading_theta), np.sin(heading_theta)], 15)
    vehicle.set_heading_theta(heading_theta, True)


def make_env(render=True, w_size=(1200, 800), env_idx=0):
    if env_idx == 0:
        env = HumanInTheLoopEnv(dict(use_render=render,
                                     map="S",
                                     environment_num=1,
                                     start_seed=312,
                                     main_exp=False,
                                     window_size=w_size,
                                     vehicle_config=dict(show_navi_mark=False, spawn_longitude=5, spawn_lateral=0)
                                     ))
        o = env.reset()
        if render:
            env.main_camera.camera.node().getLens().setAspectRatio(4 / 3)
        list(env.engine.object_manager.spawned_objects.values())[0].set_static(True)
        list(env.engine.object_manager.spawned_objects.values())[0].set_position((43, 3.5))

        # list(env.engine.object_manager.spawned_objects.values())[1].destroy()
        list(env.engine.object_manager.spawned_objects.values())[1].set_position((35, 3.5))
        return env, o
    elif env_idx == 1:
        env = HumanInTheLoopEnv(dict(use_render=render,
                                     map="SS",
                                     environment_num=1,
                                     start_seed=365,
                                     main_exp=False,
                                     traffic_density=0,
                                     window_size=w_size,
                                     vehicle_config=dict(show_navi_mark=False, spawn_longitude=70, spawn_lateral=0)
                                     ))
        o = env.reset()
        if render:
            env.main_camera.camera.node().getLens().setAspectRatio(4 / 3)
        return env, o
    elif env_idx == 2:
        env = HumanInTheLoopEnv(dict(use_render=render,
                                     map="X",
                                     environment_num=1,
                                     start_seed=365,
                                     main_exp=False,
                                     traffic_density=0,
                                     window_size=w_size,
                                     vehicle_config=dict(show_navi_mark=False)
                                     ))
        o = env.reset()
        if render:
            env.main_camera.camera.node().getLens().setAspectRatio(4 / 3)
        return env, o

    elif env_idx == 3:
        env = HumanInTheLoopEnv(dict(use_render=render,
                                     map="S",
                                     environment_num=1,
                                     start_seed=312,
                                     main_exp=False,
                                     traffic_density=0,
                                     window_size=w_size,
                                     vehicle_config=dict(show_navi_mark=False)
                                     ))
        o = env.reset()
        env.vehicle.set_position((10, 3))
        list(env.engine.object_manager.spawned_objects.values())[0].set_static(True)
        list(env.engine.object_manager.spawned_objects.values())[0].set_position((35, 6.5))
        list(env.engine.object_manager.spawned_objects.values())[0].set_heading_theta(-np.pi/4)

        list(env.engine.object_manager.spawned_objects.values())[1].destroy()
        if render:
            env.main_camera.camera.node().getLens().setAspectRatio(4 / 3)
        return env, o


def save_value(name, env_idx, heading_theta=0, cost_value=False, path="checkpoint_276\checkpoint-276", long_range=None, scale=4):
    weights = load_model(path)
    env, o = make_env(render=False, env_idx=env_idx)
    env.vehicle.body.setIntoCollideMask(BitMask32.allOff())
    start = 40 if long_range is None else int(long_range[0]/0.25)
    end = 300 if long_range is None else int(long_range[1]/0.25)
    longs = [l * 0.25 / scale for l in range(start * scale, end * scale)]
    lats = [c * 0.25 / scale for c in range(-7 * scale, 36 * scale)]
    heat_map = []
    for long in longs:
        ret = []
        for lat in lats:
            set_state(env.vehicle, (long, lat), heading_theta)
            o = env.observations["default_agent"].observe(env.vehicle)
            action = controller(o, weights)
            if cost_value:
                value = compute_cost_value(o, action, weights)
            else:
                value = compute_value(o, action, weights)
            ret.append(value[0][0])
            env.step(action)
            # env.render(text={"Value": value})
        heat_map.append(ret)
    heat_map.reverse()
    with open("{}.pkl".format(name), "wb+") as f:
        pickle.dump(heat_map, f)
    env.close()


def run_env(env_idx, w_size=(1200, 800), path="checkpoint_276\checkpoint-276", select_mode=True, freq=7):
    weights = load_model(path)
    env, o = make_env(w_size=w_size, env_idx=env_idx)
    if not select_mode:
        env.vehicle.body.setIntoCollideMask(BitMask32.allOff())
    model = BaseVehicle.model_collection["vehicle/ferra/"]
    for i in range(1000):
        o, *_ = env.step(controller(o, weights))
        if i % freq == 0 and not select_mode:
            new_p = NodePath(str(i))
            model.instanceTo(new_p)
            new_p.setPos(env.vehicle.origin.getPos())
            new_p.setH(env.vehicle.origin.getH())
            new_p.reparentTo(env.engine.pbr_worldNP)
        elif select_mode:
            env.render(text={"seed": env.current_seed})


def save_value_for_scene_2(name, cost_value=False, path="checkpoint_276\checkpoint-276", scale=4, long_range=None):
    r_0 = Road(">>", ">>>")
    r_1 = Road(">>>", "1X_0_0_")
    r_2 = Road("1X_0_0_", "1X_0_1_")
    env_idx = 2
    weights = load_model(path)
    env, o = make_env(render=False, env_idx=env_idx)
    env.vehicle.body.setIntoCollideMask(BitMask32.allOff())
    start = int(long_range[0]/0.25)
    end = int(long_range[1]/0.25)
    longs = [l * 0.25 / scale for l in range(start * scale, end * scale)]
    lats = [c * 0.25 / scale for c in range(-7 * scale, 100 * scale)]
    heat_map = []
    for long in longs:
        ret = []
        for lat in lats:
            set_state(env.vehicle, (long, lat), 0)
            action = controller(o, weights)
            env.step(action)
            o = env.observations["default_agent"].observe(env.vehicle)
            if cost_value:
                value = compute_cost_value(o, action, weights)
            else:
                value = compute_value(o, action, weights)
            ret.append(value[0][0])

            # env.render(text={"Value": value})
        heat_map.append(ret)
    heat_map.reverse()
    with open("{}.pkl".format(name), "wb+") as f:
        pickle.dump(heat_map, f)
    env.close()

if __name__ == "__main__":
    # run_env(w_size=(1600, 1200), env_idx=0, select_mode=True)
    # save_value("heat_map_vehicle_and_alert", 0)

    # run_env(w_size=(1600, 1200), env_idx=1, select_mode=False)
    # save_value("heat_map_cone", env_idx=1,scale=4,long_range=(70, 130))

    # special value save
    # run_env(w_size=(1600, 1200), env_idx=2, select_mode=False, freq=12)
    # save_value_for_scene_2("heat_map_intersection", scale=1, long_range=(50, 90))

    run_env(w_size=(1600, 1200), env_idx=3, select_mode=False, freq=7)
    # save_value("heat_map_right_barrier", env_idx=3, scale=4, long_range=(10, 80))

