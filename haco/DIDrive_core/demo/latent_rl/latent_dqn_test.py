import os
import torch
from easydict import EasyDict

from haco.DIDrive_core.envs import BenchmarkEnvWrapper
from haco.DIDrive_core.utils.others.tcp_helper import parse_carla_tcp
from haco.DIDrive_core.eval import SingleCarlaEvaluator
from ding.policy import DQNPolicy
from ding.utils import set_pkg_seed
from ding.utils.default_helper import deep_merge_dicts

from haco.DIDrive_core.demo.latent_rl.latent_rl_env import CarlaLatentEvalEnv
from haco.DIDrive_core.demo.latent_rl.model import LatentDQNRLModel

eval_config = dict(
    env=dict(
        simulator=dict(
            town='Town01',
            disable_two_wheels=True,
            verbose=False,
            planner=dict(
                type='lbc',
                resolution=2.5,
                threshold_before=9,
                threshold_after=1.5,
            ),
            obs=(
                dict(
                    name='birdview',
                    type='bev',
                    size=[320, 320],
                    pixels_per_meter=5,
                    pixels_ahead_vehicle=100,
                ),
            )
        ),
        discrete_action=True,
        discrete_dim=10,
        visualize=dict(type='birdview', outputs=['show']),
        wrapper=dict(
            suite='FullTown02-v1',
        ),
    ),
    policy=dict(
        cuda=True,
        ckpt_path='',
        model=dict(action_shape=100),
        eval=dict(
            evaluator=dict(
                render=True,
                transform_obs=True,
            ),
        ),
    ),
    server=[dict(
        carla_host='localhost',
        carla_ports=[9000, 9002, 2]
    )],
)

main_config = EasyDict(eval_config)


def main(cfg, seed=0):
    cfg.policy = deep_merge_dicts(DQNPolicy.default_config(), cfg.policy)

    tcp_list = parse_carla_tcp(cfg.server)
    assert len(tcp_list) > 0, "No Carla server found!"
    host, port = tcp_list[0]

    carla_env = BenchmarkEnvWrapper(CarlaLatentEvalEnv(cfg.env, host, port), cfg.env.wrapper)
    carla_env.seed(seed)
    set_pkg_seed(seed)
    model = LatentDQNRLModel(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)

    if cfg.policy.ckpt_path != '':
        state_dict = torch.load(cfg.policy.ckpt_path, map_location='cpu')
        policy.eval_mode.load_state_dict(state_dict)
    evaluator = SingleCarlaEvaluator(cfg.policy.eval.evaluator, carla_env, policy.eval_mode)
    evaluator.eval()
    evaluator.close()


if __name__ == '__main__':
    main(main_config)
