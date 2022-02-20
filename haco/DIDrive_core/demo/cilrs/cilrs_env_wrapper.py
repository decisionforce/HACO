from typing import Any

import PIL
import numpy as np
from haco.DIDrive_core.envs import CarlaEnvWrapper


class CILRSEnvWrapper(CarlaEnvWrapper):
    config = dict(
        scale=1,
        crop=256,
        speed_factor=25.,
    )

    def _get_obs(self, obs):
        new_obs = {
            'command': obs['command'],
            'speed': np.float32(obs['speed'] / self._cfg.speed_factor),
            'tick': obs['tick'],
        }
        rgb = obs['rgb']
        im = PIL.Image.fromarray(rgb)
        (width, height) = (int(im.width // self._cfg.scale), int(im.height // self._cfg.scale))
        im_resized = im.resize((width, height))
        image = np.asarray(im_resized)
        start_x = height // 2 - self._cfg.crop // 2
        start_y = width // 2 - self._cfg.crop // 2
        cropped_image = image[start_x:start_x + self._cfg.crop, start_y:start_y + self._cfg.crop]
        new_obs['rgb'] = cropped_image
        return new_obs

    def reset(self, *args, **kwargs) -> Any:
        obs = super().reset(*args, **kwargs)
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
