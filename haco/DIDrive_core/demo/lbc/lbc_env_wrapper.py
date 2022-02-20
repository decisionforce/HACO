import torch
import numpy as np
from typing import Dict, Any
import math

from haco.DIDrive_core.envs import CarlaEnvWrapper
from haco.DIDrive_core.utils.model_utils import common


class LBCEnvWrapper(CarlaEnvWrapper):

    def _get_obs(self, obs):
        new_obs = {
            'command': obs['command'],
            'speed': np.float32(obs['speed'] / 3.6),
        }
        if 'rgb' in obs:
            new_obs['rgb'] = obs['rgb'] / 255.
        elif 'birdview' in obs:
            birdview = obs['birdview'][..., :7]
            birdview = common.crop_birdview(birdview, dx=-10)
            new_obs['birdview'] = birdview
        return new_obs

    def reset(self, *args, **kwargs) -> Any:
        obs = super().reset()
        obs_out = self._get_obs(obs)
        return obs_out

    def step(self, action):
        action = {
            'steer': action['steer'],
            'throttle': action['throttle'],
            'brake': action['brake'],
        }
        timestep = super().step(action)
        obs = timestep.obs
        obs_out = self._get_obs(obs)
        timestep = timestep._replace(obs=obs_out)
        return timestep
