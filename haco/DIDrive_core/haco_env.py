# import evdev
import gym
import numpy as np
import pygame
from easydict import EasyDict
# from evdev import ecodes, InputDevice

from haco.DIDrive_core.envs.simple_carla_env import SimpleCarlaEnv
from demo.simple_rl.env_wrapper import ContinuousBenchmarkEnvWrapper
from demo.simple_rl.sac_train import compile_config


def safe_clip(array, min_val, max_val):
    array = np.nan_to_num(array.astype(np.float64), copy=False, nan=0.0, posinf=max_val, neginf=min_val)
    return np.clip(array, min_val, max_val).astype(np.float64)


train_config = dict(
    env=dict(
        collector_env_num=1,
        evaluator_env_num=0,
        simulator=dict(
            town='Town01',
            disable_two_wheels=True,
            verbose=False,
            waypoint_num=32,
            planner=dict(
                type='behavior',
                resolution=1,
            ),
            obs=(
                dict(
                    name='rgb',
                    type='rgb',
                    size=[2400, 1600],
                    position=[-5.5, 0, 2.8],
                    rotation=[-15, 0, 0],
                ),
                dict(
                    name='birdview',
                    type='bev',
                    size=[42, 42],
                    pixels_per_meter=2,
                    pixels_ahead_vehicle=16,
                ),
                # dict(
                #     name='birdview',
                #     type='bev',
                #     size=[32, 32],
                #     pixels_per_meter=1,
                #     pixels_ahead_vehicle=14,
                # ),
            )
        ),
        col_is_failure=True,
        stuck_is_failure=False,
        wrong_direction_is_failure=False,
        off_route_is_failure=False,
        off_road_is_failure=True,
        ignore_light=True,
        visualize=dict(
            type='rgb',
            outputs=['show']
        ),
        manager=dict(
            collect=dict(
                auto_reset=True,
                shared_memory=False,
                context='spawn',
                max_retry=1,
            ),
            eval=dict()
        ),
        wrapper=dict(
            # Collect and eval suites for training
            collect=dict(suite='train_ft'),

        ),
    ),
)


class SteeringWheelController:
    RIGHT_SHIFT_PADDLE = 4
    LEFT_SHIFT_PADDLE = 5
    STEERING_MAKEUP = 1.5

    def __init__(self):
        pygame.display.init()
        pygame.joystick.init()
        assert pygame.joystick.get_count() > 0, "Please connect joystick or use keyboard input"
        print("Successfully Connect your Joystick!")

        # ffb_device = evdev.list_devices()[0]
        # self.ffb_dev = InputDevice(ffb_device)

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        self.right_shift_paddle = False
        self.left_shift_paddle = False

        self.button_circle = False
        self.button_rectangle = False
        self.button_triangle = False
        self.button_x = False

        self.button_up = False
        self.button_down = False
        self.button_right = False
        self.button_left = False

    def process_input(self, speed):
        pygame.event.pump()
        steering = -self.joystick.get_axis(0)
        throttle = (1 - self.joystick.get_axis(1)) / 2
        brake = (1 - self.joystick.get_axis(3)) / 2
        offset = 30
        # val = int(65535 * (speed + offset) / (120 + offset))
        # self.ffb_dev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, val)
        self.right_shift_paddle = True if self.joystick.get_button(self.RIGHT_SHIFT_PADDLE) else False
        self.left_shift_paddle = True if self.joystick.get_button(self.LEFT_SHIFT_PADDLE) else False

        self.left_shift_paddle = True if self.joystick.get_button(self.LEFT_SHIFT_PADDLE) else False
        self.left_shift_paddle = True if self.joystick.get_button(self.LEFT_SHIFT_PADDLE) else False

        self.button_circle = True if self.joystick.get_button(2) else False
        self.button_rectangle = True if self.joystick.get_button(1) else False
        self.button_triangle = True if self.joystick.get_button(3) else False
        self.button_x = True if self.joystick.get_button(0) else False

        hat = self.joystick.get_hat(0)
        self.button_up = True if hat[-1] == 1 else False
        self.button_down = True if hat[-1] == -1 else False
        self.button_left = True if hat[0] == -1 else False
        self.button_right = True if hat[0] == 1 else False

        return [-steering * self.STEERING_MAKEUP, (throttle - brake)]


class HACOEnv(ContinuousBenchmarkEnvWrapper):
    def __init__(self, config=None, eval=False, port=9000):
        main_config = EasyDict(train_config)
        self.eval = eval
        if eval:
            train_config["env"]["wrapper"]["collect"]["suite"] = 'FullTown02-v1'
        cfg = compile_config(main_config)
        super(HACOEnv, self).__init__(SimpleCarlaEnv(cfg.env, "localhost", port, None), cfg.env.wrapper.collect)
        try:
            self.controller = SteeringWheelController() if not eval else None
        except:
            self.controller = None
        self.last_takeover = False
        self.total_takeover_cost = 0
        self.episode_reward = 0

    def step(self, action):
        if self.controller is not None:
            human_action = self.controller.process_input(self.env._simulator_databuffer['state']['speed'] * 3.6)
            takeover = self.controller.left_shift_paddle or self.controller.right_shift_paddle
        else:
            human_action = [0, 0]
            takeover = False
        o, r, d, info = super(HACOEnv, self).step(human_action if takeover else action)
        self.episode_reward += r
        if not self.last_takeover and takeover:
            cost = self.get_takeover_cost(human_action, action)
            self.total_takeover_cost += cost
            info["takeover_cost"] = cost
        else:
            info["takeover_cost"] = 0

        info["takeover"] = takeover
        info["total_takeover_cost"] = self.total_takeover_cost
        info["raw_action"] = action if not takeover else human_action
        self.last_takeover = takeover

        info["velocity"] = self.env._simulator_databuffer['state']['speed']
        info["steering"] = info["raw_action"][0]
        info["acceleration"] = info["raw_action"][1]
        info["step_reward"] = r
        info["cost"] = self.native_cost(info)
        info["native_cost"] = info["cost"]
        info["out_of_road"] = info["off_road"]
        info["crash"] = info["collided"]
        info["arrive_dest"] = info["success"]
        info["episode_length"] = info["tick"]
        info["episode_reward"] = self.episode_reward
        if not self.eval:
            self.render()
        return o, r[0], d, info

    def native_cost(self, info):
        if info["off_route"] or info["off_road"] or info["collided"] or info["wrong_direction"]:
            return 1
        else:
            return 0

    def get_takeover_cost(self, human_action, agent_action):
        takeover_action = safe_clip(np.array(human_action), -1, 1)
        agent_action = safe_clip(np.array(agent_action), -1, 1)
        # cos_dist = (agent_action[0] * takeover_action[0] + agent_action[1] * takeover_action[1]) / 1e-6 +(
        #         np.linalg.norm(takeover_action) * np.linalg.norm(agent_action))

        multiplier = (agent_action[0] * takeover_action[0] + agent_action[1] * takeover_action[1])
        divident = np.linalg.norm(takeover_action) * np.linalg.norm(agent_action)
        if divident < 1e-6:
            cos_dist = 1.0
        else:
            cos_dist = multiplier / divident
        return 1 - cos_dist

    def reset(self, *args, **kwargs):
        self.last_takeover = False
        self.total_takeover_cost = 0
        self.episode_reward = 0
        return super(HACOEnv, self).reset()

    @property
    def action_space(self):
        return gym.spaces.Box(-1.0, 1.0, shape=(2,))

    @property
    def observation_space(self):
        return gym.spaces.Dict({"birdview": gym.spaces.Box(low=0, high=1, shape=(42, 42, 5), dtype=np.uint8),
                                "speed": gym.spaces.Box(-10., 10.0, shape=(1,))})


if __name__ == "__main__":
    env = HACOEnv()
    o = env.reset()

    while True:
        if not env.observation_space.contains(o):
            print(o)
        o, r, d, i = env.step([0., -0.0])

        if d:
            env.reset()
