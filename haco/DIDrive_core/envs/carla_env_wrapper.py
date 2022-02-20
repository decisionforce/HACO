import copy
from itertools import product
from typing import Any, Dict, Optional

import gym
import numpy as np
from ding.envs.env.base_env import BaseEnvTimestep, BaseEnvInfo
from ding.torch_utils.data_helper import to_ndarray
from easydict import EasyDict
from haco.DIDrive_core.data.benchmark import ALL_SUITES
from haco.DIDrive_core.eval.carla_benchmark_evaluator import get_suites_list, read_pose_txt, get_benchmark_dir
from haco.DIDrive_core.utils.others.config_helper import deep_merge_dicts

from .base_carla_env import BaseCarlaEnv


class CarlaEnvWrapper(gym.Wrapper):
    """
    Environment wrapper to make ``gym.Env`` align with DI-engine definitions, so as to use utilities in DI-engine.
    It changes ``step``, ``reset`` and ``info`` method of ``gym.Env``, while others are straightly delivered.

    :Arguments:
        - env (BaseCarlaEnv): The environment to be wrapped.
        - cfg (Dict): Config dict.

    :Interfaces: reset, step, info, render, seed, close
    """

    config = dict()

    def __init__(self, env: BaseCarlaEnv, cfg: Dict = None, **kwargs) -> None:
        if cfg is None:
            self._cfg = self.__class__.default_config()
        elif 'cfg_type' not in cfg:
            self._cfg = self.__class__.default_config()
            self._cfg = deep_merge_dicts(self._cfg, cfg)
        else:
            self._cfg = cfg
        self.env = env

    def reset(self, *args, **kwargs) -> Any:
        """
        Wrapper of ``reset`` method in env. The observations are converted to ``np.ndarray`` and final reward
        are recorded.

        :Returns:
            Any: Observations from envirionment
        """
        obs = self.env.reset(*args, **kwargs)
        obs = to_ndarray(obs)
        self._final_eval_reward = 0.0
        return obs

    def step(self, action: Any = None) -> BaseEnvTimestep:
        """
        Wrapper of ``step`` method in env. This aims to convert the returns of ``gym.Env`` step method into
        that of ``ding.envs.BaseEnv``, from ``(obs, reward, done, info)`` tuple to a ``BaseEnvTimestep``
        namedtuple defined in DI-engine. It will also convert actions, observations and reward into
        ``np.ndarray``, and check legality if action contains control signal.

        :Arguments:
            - action (Any, optional): Actions sent to env. Defaults to None.

        :Returns:
            BaseEnvTimestep: DI-engine format of env step returns.
        """
        action = to_ndarray(action)
        if action is not None:
            for key in ['throttle', 'brake']:
                if key in action:
                    np.clip(action[key], 0, 1)
            if 'steer' in action:
                np.clip(action['steer'], -1, 1)

        obs, rew, done, info = self.env.step(action)
        self._final_eval_reward += rew
        obs = to_ndarray(obs)
        rew = to_ndarray([rew], dtype=np.float32)
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> BaseEnvInfo:
        """
        Interface of ``info`` method to suit DI-engine format env.
        It returns a namedtuple ``BaseEnvInfo`` defined in DI-engine
        which contains information about observation, action and reward space.

        :Returns:
            BaseEnvInfo: Env information instance defined in DI-engine.
        """
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        return BaseEnvInfo(agent_num=1, obs_space=obs_space, act_space=act_space, use_wrappers=None)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        self.env = gym.wrappers.Monitor(self.env, self._replay_path, video_callable=lambda episode_id: True, force=True)

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(cls.config)
        cfg.cfg_type = cls.__name__ + 'Config'
        return copy.deepcopy(cfg)

    def __repr__(self) -> str:
        return repr(self.env)

    def render(self):
        self.env.render()


class BenchmarkEnvWrapper(CarlaEnvWrapper):
    """
    Environment Wrapper for Carla Benchmark suite evaluations. It wraps an environment with Benchmark
    suite so that the env will always run with a benchmark suite setting. It has 2 mode to get reset
    params in a suite: 'random' will randomly get reset param, 'order' will get all reset params in
    order.

    :Arguments:
        - env (BaseCarlaEnv): The environment to be wrapped.
        - cfg (Dict): Config dict.
    """

    config = dict(
        suite='FullTown01-v0',
        benchmark_dir=None,
        mode='random',
    )

    def __init__(self, env: BaseCarlaEnv, cfg: Dict, **kwargs) -> None:
        super().__init__(env, cfg=cfg, **kwargs)
        suite = self._cfg.suite
        benchmark_dir = self._cfg.benchmark_dir
        self._mode = self._cfg.mode
        if benchmark_dir is None:
            benchmark_dir = get_benchmark_dir()
        assert self._mode in ['random', 'order'], self._mode
        self._param = dict()
        suite_list = get_suites_list(suite)

        self._reset_param_list = []
        for suite in suite_list:
            args, kwargs = ALL_SUITES[suite]
            assert len(args) == 0
            reset_params = kwargs.copy()
            poses_txt = reset_params.pop('poses_txt')
            weathers = [8]
            pose_pairs = read_pose_txt(benchmark_dir, poses_txt)
            for (start, end), weather in product(pose_pairs, weathers):
                param = reset_params.copy()
                param['start'] = start
                param['end'] = end
                param['weather'] = weather
                param['col_is_failure'] = True
                self._reset_param_list.append(param)
        self._reset_param_index = 0

    def reset(self, *args, **kwargs) -> Any:
        """
        Wrapped ``reset`` method for env. it will ignore all incoming arguments and choose one
        from suite reset parameters according to config.

        :Returns:
            Any: Returns of Env `reset` method.
        """
        if self._mode == 'random':
            self._param = np.random.choice(self._reset_param_list)
        elif self._mode == 'order':
            self._param = self._reset_param_list[self._reset_param_index]
            self._reset_param_index + 1
            if self._reset_param_index >= len(self._reset_param_list):
                self._reset_param_index = 0
        return super().reset(**self._param)

    def step(self, action: Dict) -> Any:
        """
        Wrapped ``step`` method for Env. It will add a print log when the env is done.

        :Arguments:
            - action (Any): Actions sent to env.

        :Returns:
            Any: Env step result.
        """
        timestep = super().step(action)
        done = timestep.done
        info = timestep.info
        if done:
            done_tick = info['tick']
            done_reward = info['final_eval_reward']
            if info['success']:
                done_state = 'Success'
            elif info['collided']:
                done_state = "Collided"
            elif info['wrong_direction']:
                done_state = "Wrong Direction"
            elif info['off_road']:
                done_state = "Off road"
            elif info['stuck']:
                done_state = "Stuck"
            elif info['timeout']:
                done_state = "Timeout"
            else:
                done_state = 'None'
            print(
                "[ENV] {} done with tick: {}, state: {}, reward: {}".format(
                    repr(self.env), done_tick, done_state, done_reward
                )
            )
        return timestep


# TODO: complete scenario env wrapper
class ScenarioEnvWrapper(CarlaEnvWrapper):
    config = dict()

    def __init__(self, env: BaseCarlaEnv, cfg: Dict, **kwargs) -> None:
        super().__init__(env, cfg=cfg, **kwargs)
