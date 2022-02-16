import copy
import json
import math
import numbers
from collections import defaultdict

import numpy as np
import yaml
from gym import Wrapper


class SafeFallbackEncoder(json.JSONEncoder):
    def __init__(self, nan_str="null", **kwargs):
        super(SafeFallbackEncoder, self).__init__(**kwargs)
        self.nan_str = nan_str

    def default(self, value):
        try:
            if np.isnan(value):
                return self.nan_str

            if (type(value).__module__ == np.__name__
                    and isinstance(value, np.ndarray)):
                return value.tolist()

            if issubclass(type(value), numbers.Integral):
                return int(value)
            if issubclass(type(value), numbers.Number):
                return float(value)

            return super(SafeFallbackEncoder, self).default(value)

        except Exception:
            return str(value)  # give up, just stringify it (ok for logs)


def pretty_print(result):
    result = result.copy()
    result.update(config=None)  # drop config from pretty print
    result.update(hist_stats=None)  # drop hist_stats from pretty print
    out = {}
    for k, v in result.items():
        if v is not None:
            out[k] = v

    cleaned = json.dumps(out, cls=SafeFallbackEncoder)
    return yaml.safe_dump(json.loads(cleaned), default_flow_style=False)


def norm(a, b):
    return math.sqrt(a ** 2 + b ** 2)


def merge_dicts(d1, d2):
    """
    Args:
        d1 (dict): Dict 1.
        d2 (dict): Dict 2.

    Returns:
         dict: A new dict that is d1 and d2 deep merged.
    """
    merged = copy.deepcopy(d1)
    deep_update(merged, d2, True, [])
    return merged


def deep_update(original,
                new_dict,
                new_keys_allowed=False,
                allow_new_subkey_list=None,
                override_all_if_type_changes=None):
    """Updates original dict with values from new_dict recursively.

    If new key is introduced in new_dict, then if new_keys_allowed is not
    True, an error will be thrown. Further, for sub-dicts, if the key is
    in the allow_new_subkey_list, then new subkeys can be introduced.

    Args:
        original (dict): Dictionary with default values.
        new_dict (dict): Dictionary with values to be updated
        new_keys_allowed (bool): Whether new keys are allowed.
        allow_new_subkey_list (Optional[List[str]]): List of keys that
            correspond to dict values where new subkeys can be introduced.
            This is only at the top level.
        override_all_if_type_changes(Optional[List[str]]): List of top level
            keys with value=dict, for which we always simply override the
            entire value (dict), iff the "type" key in that value dict changes.
    """
    allow_new_subkey_list = allow_new_subkey_list or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            raise Exception("Unknown config parameter `{}` ".format(k))

        # Both orginal value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if k in override_all_if_type_changes and \
                    "type" in value and "type" in original[k] and \
                    value["type"] != original[k]["type"]:
                original[k] = value
            # Allowed key -> ok to add new subkeys.
            elif k in allow_new_subkey_list:
                deep_update(original[k], value, True)
            # Non-allowed key.
            else:
                deep_update(original[k], value, new_keys_allowed)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original


class RecorderEnv(Wrapper):
    """Modify from RLLib callbacks"""
    _default_eval_config = dict(neighbours_distance=20)

    EPISODE_END = -1

    def __init__(self, env, eval_config=None):
        super(RecorderEnv, self).__init__(env)
        eval_config = deep_update(self._default_eval_config, eval_config or {})
        self.eval_config = eval_config

    def reset(self, force_seed=None):
        o = super(RecorderEnv, self).reset(force_seed=force_seed)
        self.episode_step = 0
        return o

    def step(self, *args, **kwargs):
        o, r, d, i = super(RecorderEnv, self).step(*args, **kwargs)
        self.on_episode_step(r, i)
        if d:
            self.on_episode_end(i)
        return o, r, d, i

    def on_episode_start(self):
        self.user_data = defaultdict(list)
        self.episode_step = 0

    def on_episode_step(self, r, info):
        if self.episode_step == 0:
            self.on_episode_start()

        if info:
            self.user_data["velocity"].append(info["velocity"])
            self.user_data["steering"].append(info["steering"])
            self.user_data["step_reward"].append(info["step_reward"])
            self.user_data["acceleration"].append(info["acceleration"])
            self.user_data["cost"].append(info["cost"])
            self.user_data["episode_length"].append(info["episode_length"])
            self.user_data["episode_reward"].append(info["episode_reward"])
            self.user_data["success"].append(info["arrive_dest"])

        # Count
        self.episode_step += 1

    def on_episode_end(self, i):
        pass

    def get_episode_result(self):
        ret = {}
        ep_len = ret["episode_length"] = len(self.user_data["cost"])
        ret["velocity_step_mean"] = np.mean(self.user_data["velocity"]) if ep_len > 0 else 0.0
        ret["success"] = self.user_data["success"][-1] if ep_len > 0 else 0.0
        ret["episode_reward"] = sum(self.user_data["step_reward"]) if ep_len > 0 else 0.0
        ret["episode_cost"] = sum(self.user_data["cost"]) if ep_len > 0 else 0.0
        return ret
