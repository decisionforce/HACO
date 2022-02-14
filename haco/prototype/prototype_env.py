import os
from drivingforce.process.vis_model_utils import expert_action_prob

import gym
import numpy as np

from drivingforce.expert_in_the_loop.expert_guided_env import ExpertGuidedEnv

expert_value_weights = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "expert.npz")


class PrototypeEnv(ExpertGuidedEnv):
    INTENTION_LANE_KEEP = [0., 1., 0.]
    # INTENTION_DEC = 1
    INTENTION_LEFT_LANE_CHANGE = [1., 0., 0.]
    INTENTION_RIGHT_LANE_CHANGE = [0., 0., 1.]

    def default_config(self):
        """
        Train/Test set both contain 10 maps
        :return: PGConfig
        """
        config = super(PrototypeEnv, self).default_config()
        config.update(dict(
            environment_num=1,
            start_seed=13,
            map="S",
            expert_value_weights=expert_value_weights,
            traffic_density=0,
            vehicle_config=dict(spawn_longitude=55, use_saver=True, free_level=0.95)
        ), allow_add_new_key=True)
        return config

    def reset(self, *args, **kwargs):
        self.total_takeover_cost = 0
        ret = super(PrototypeEnv, self).reset(*args, **kwargs)
        self.vehicle.update_config({"max_speed": 40})
        self.agent_intention = self.INTENTION_LANE_KEEP
        self.last_intention = self.INTENTION_LANE_KEEP
        self.external_intention = self.INTENTION_LANE_KEEP
        self.last_same_intention = True
        return ret

    def step(self, actions):
        self.last_intention = self.agent_intention
        action, self.agent_intention = actions[:2], actions[2:]
        o, r, d, i = super(PrototypeEnv, self).step(action)
        # print(self.agent_intention)
        if self.engine is not None and self.config["use_render"]:
            env=self
            info = i
            self.render(text={"seed": env.current_seed, "lane_index": env.vehicle.lane_index,
                         "longitude": env.vehicle.lane.local_coordinates(env.vehicle.position)[0],
                         "human_intention": env.human_intention,
                         "agent_intention": env.agent_intention,
                         "takeover": info["takeover"],
                         "takeover_cost": info["takeover_cost"],
                         "total_takeover_cost": info["total_takeover_cost"]})
        return o, r, d, i

    def extra_step_info(self, step_infos):
        step_infos["native_cost"] = step_infos["cost"]
        self.total_native_cost += step_infos["native_cost"]
        step_infos["total_native_cost"] = self.total_native_cost
        if (np.array(self.human_intention) != np.array(self.agent_intention)).any():
            # intention not align or low level mismatch, give cost
            self.total_takeover_cost += 1
            step_infos["intention_violation"] = 1
            this_frame_same = False
        else:
            self.total_takeover_cost += 0
            step_infos["intention_violation"] = 0
            this_frame_same = True

        step_infos["takeover_cost"] = 1 if self.last_same_intention and not this_frame_same else 0
        self.last_same_intention = this_frame_same
        step_infos["total_takeover_cost"] = self.total_takeover_cost
        step_infos["native_cost"] = step_infos["cost"]
        step_infos["total_native_cost"] = self.episode_cost
        step_infos["human_intention"] = self.human_intention
        step_infos["newbie_intention"] = self.agent_intention
        return step_infos

    def saver(self, v_id: str, actions):
        """
        Action prob takeover
        """
        if self.config["rule_takeover"]:
            return self.rule_takeover(v_id, actions)
        vehicle = self.vehicles[v_id]
        action = actions
        steering = action[0]
        throttle = action[1]
        self.state_value = 0
        pre_save = vehicle.takeover
        if vehicle.config["use_saver"] or vehicle.expert_takeover:
            # saver can be used for human or another AI
            free_level = vehicle.config["free_level"] if not vehicle.expert_takeover else 1.0
            obs = self.expert_observation.observe(vehicle)
            try:
                saver_a, a_0_p, a_1_p = expert_action_prob(action, obs, self.expert_weights,
                                                           deterministic=vehicle.config["expert_deterministic"])
            except ValueError:
                print("Expert can not takeover, due to observation space mismathing!")
                saver_a = action
            else:
                if free_level <= 1e-3:
                    steering = saver_a[0]
                    throttle = saver_a[1]
                elif free_level > 1e-3:
                    if (np.array(self.human_intention) != np.array(self.agent_intention)).any() or (a_0_p * a_1_p < 1 - \
                            vehicle.config["free_level"]):
                        # intention not align takeover or action mismatch
                        steering, throttle = saver_a[0], saver_a[1]

        # indicate if current frame is takeover step
        vehicle.takeover = True if action[0] != steering or action[1] != throttle else False
        saver_info = {
            "takeover_start": True if not pre_save and vehicle.takeover else False,
            "takeover_end": True if pre_save and not vehicle.takeover else False,
            "takeover": vehicle.takeover if pre_save else False
        }
        if saver_info["takeover"]:
            saver_info["raw_action"] = [steering, throttle]
        return (steering, throttle) if saver_info["takeover"] else action, saver_info

    @property
    def human_intention(self):
        long, _ = self.vehicle.lane.local_coordinates(self.vehicle.position)
        if 20 < long < 30:
            return self.INTENTION_RIGHT_LANE_CHANGE
        return self.INTENTION_LANE_KEEP

    @property
    def action_space(self):
        if self.engine is None:
            return super(PrototypeEnv, self).action_space
        else:
            return gym.spaces.Box(-1, 1, (5,))

    def need_external_intention(self):
        # if self.engine.get_policy(self.vehicle.id).controller.button_x:
        #     self.human_intention = self.INTENTION_DEC
        self.engine.accept("a", self._left_change_lane)
        self.engine.accept("w", self._lane_keep)
        self.engine.accept("d", self._right_change_lane)

    def _right_change_lane(self):
        self.external_intention = self.INTENTION_RIGHT_LANE_CHANGE

    def _lane_keep(self):
        self.external_intention = self.INTENTION_LANE_KEEP

    def _left_change_lane(self):
        self.external_intention = self.INTENTION_LEFT_LANE_CHANGE


if __name__ == "__main__":
    env = PrototypeEnv({"use_render": True, "manual_control": False, "vehicle_config":{"free_level":0}})
    env.reset()
    while True:
        *_, d, info = env.step([0, 0, 0, 1.0, 0])
        env.render(text={"seed": env.current_seed, "lane_index": env.vehicle.lane_index,
                         "longitude": env.vehicle.lane.local_coordinates(env.vehicle.position)[0],
                         "human_intention": env.human_intention,
                         "agent_intention": env.agent_intention,
                         "takeover": info["takeover"],
                         "takeover_cost": info["takeover_cost"],
                         "total_takeover_cost": info["total_takeover_cost"]})
        if env.human_intention == env.INTENTION_RIGHT_LANE_CHANGE:
            print(info["raw_action"])
            d=True
        # if env.human_intention == env.INTENTION_RIGHT_LANE_CHANGE:
        #     assert info["takeover"]
        if d:
            env.reset()
