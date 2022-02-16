from typing import Dict, Union, Optional, Type, List

import numpy as np
from ray.rllib.agents.ppo.ppo import validate_config as original_validate_config, PPOTrainer, \
    DEFAULT_CONFIG as ppo_default_config, warn_about_bad_reward_scales, UpdateKL
from ray.rllib.agents.ppo.ppo_tf_policy import setup_mixins, ValueNetworkMixin, KLCoeffMixin, EntropyCoeffSchedule, \
    PPOTFPolicy, kl_and_loss_stats, LearningRateSchedule, postprocess_ppo_gae
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.evaluation import postprocessing as rllib_post
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.common import _get_shared_metrics
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches, StandardizeFields, SelectExperiences
from ray.rllib.execution.train_ops import TrainOneStep, TrainTFMultiGPU
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, MultiAgentBatch
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.typing import AgentID, TensorType, TrainerConfigDict
from ray.tune.utils.util import merge_dicts
from ray.util.iter import LocalIterator

from haco.ppo_lag.ppo_lag_model import CostValueNetwork, CostValueNetworkMixin

if hasattr(rllib_post, "discount_cumsum"):
    discount = rllib_post.discount_cumsum
else:
    discount = rllib_post.discount
Postprocessing = rllib_post.Postprocessing
tf, _, _ = try_import_tf()

COST_ADVANTAGE = "cost_advantage"
COST = "cost"
COST_LIMIT = "cost_limit"
COST_TARGET = "cost_target"
COST_VALUES = "cost_values"
PENALTY_LR = "penalty_lr"

PPO_LAG_CONFIG = merge_dicts(ppo_default_config, {
    COST_LIMIT: 0.0,  # Or 25, 50.
    PENALTY_LR: 1e-2,
    "batch_mode": "complete_episodes",
    "lr": 1e-4,
    "num_sgd_iter": 10,
    "train_batch_size": 30000,
    "num_workers": 5,
})


def compute_cost_advantages(rollout: SampleBatch, last_r: float, gamma: float = 0.9, lambda_: float = 1.0):
    vpred_t = np.concatenate([rollout[COST_VALUES], np.array([last_r])])
    delta_t = (rollout[COST] + gamma * vpred_t[1:] - vpred_t[:-1])
    rollout[COST_ADVANTAGE] = discount(delta_t, gamma * lambda_)
    rollout[COST_TARGET] = (rollout[COST_ADVANTAGE] + rollout[COST_VALUES]).copy().astype(np.float32)
    rollout[COST_ADVANTAGE] = rollout[COST_ADVANTAGE].copy().astype(np.float32)
    return rollout


def postprocess_ppo_cost(policy: Policy, sample_batch: SampleBatch) -> SampleBatch:
    # Trajectory is actually complete -> last r=0.0.
    if sample_batch[SampleBatch.DONES][-1]:
        last_r = 0.0
    # Trajectory has been truncated -> last r=VF estimate of last obs.
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append(sample_batch["state_out_{}".format(i)][-1])
        last_r = policy._cost_value(
            sample_batch[SampleBatch.NEXT_OBS][-1], sample_batch[SampleBatch.ACTIONS][-1], sample_batch[COST][-1],
            *next_state
        )

    # Adds the policy logits, VF preds, and advantages to the batch,
    # using GAE ("generalized advantage estimation") or not.
    batch = compute_cost_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"])
    return batch


def post_process_fn(policy: Policy,
                    sample_batch: SampleBatch,
                    other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
                    episode: Optional[MultiAgentEpisode] = None) -> SampleBatch:
    # Put the actions to batch
    infos = sample_batch.get(SampleBatch.INFOS)
    if infos is not None:
        sample_batch[COST] = np.array([info["cost"] for info in infos])
    else:  # Fill the elements if not initialized
        sample_batch[COST] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
    sample_batch = postprocess_ppo_cost(policy, sample_batch)
    sample_batch = postprocess_ppo_gae(policy, sample_batch, other_agent_batches, episode)
    return sample_batch


def ppo_lag_surrogate_loss(
        policy: Policy, model: ModelV2, dist_class: Type[TFActionDistribution],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for Proximal Policy Objective.

    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    # Setup the lambda multiplier.
    # _ = train_batch[COST]  # touch
    # _ = train_batch[SampleBatch.DONES]  # touch
    # _ = train_batch[SampleBatch.NEXT_OBS]  # touch
    # _ = train_batch[COST_ADVANTAGE]  # touch
    # _ = train_batch[COST_TARGET]  # touch
    penalty_init = 1.0
    # penalty_init = 0.1
    with tf.variable_scope('penalty'):
        param_init = np.log(max(np.exp(penalty_init) - 1, 1e-8))
        penalty_param = tf.get_variable(
            'penalty_param',
            initializer=float(param_init),
            trainable=True,
            dtype=tf.float32
        )
    penalty = tf.nn.softplus(penalty_param)
    policy._penalty = penalty
    policy._penalty_param = penalty_param

    logits, state = model.from_batch(train_batch)
    curr_action_dist = dist_class(logits, model)

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])

        def reduce_mean_valid(t):
            return tf.reduce_mean(tf.boolean_mask(t, mask))

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = tf.reduce_mean

    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)
    logp_ratio = tf.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) -
        train_batch[SampleBatch.ACTION_LOGP])
    action_kl = prev_action_dist.kl(curr_action_dist)
    mean_kl = reduce_mean_valid(action_kl)

    curr_entropy = curr_action_dist.entropy()
    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = tf.minimum(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES] * tf.clip_by_value(
            logp_ratio, 1 - policy.config["clip_param"],
                        1 + policy.config["clip_param"]))

    cost_adv = train_batch[COST_ADVANTAGE]
    surrogate_cost = cost_adv * tf.clip_by_value(logp_ratio, 0., 1 + policy.config["clip_param"])

    mean_policy_loss = reduce_mean_valid(-surrogate_loss)

    mean_cost_loss = reduce_mean_valid(surrogate_cost)

    cost_value_loss = tf.math.square(model.get_cost_value() - train_batch[COST_TARGET])

    if policy.config["use_gae"]:
        prev_value_fn_out = train_batch[SampleBatch.VF_PREDS]
        value_fn_out = model.value_function()
        vf_loss1 = tf.math.square(value_fn_out - train_batch[Postprocessing.VALUE_TARGETS])
        vf_clipped = prev_value_fn_out + tf.clip_by_value(
            value_fn_out - prev_value_fn_out, -policy.config["vf_clip_param"],
            policy.config["vf_clip_param"])
        vf_loss2 = tf.math.square(vf_clipped -
                                  train_batch[Postprocessing.VALUE_TARGETS])
        vf_loss = tf.maximum(vf_loss1, vf_loss2)
        mean_vf_loss = reduce_mean_valid(vf_loss)
        total_loss = reduce_mean_valid(
            -surrogate_loss + policy.kl_coeff * action_kl +
            policy.config["vf_loss_coeff"] * vf_loss -
            policy.entropy_coeff * curr_entropy
        ) + penalty * mean_cost_loss
        total_loss = total_loss / (1 + penalty)

    else:
        raise ValueError()
        mean_vf_loss = tf.constant(0.0)
        total_loss = reduce_mean_valid(-surrogate_loss +
                                       policy.kl_coeff * action_kl -
                                       policy.entropy_coeff * curr_entropy)

    policy._mean_cost_loss = mean_cost_loss
    policy._mean_cost_value_loss = reduce_mean_valid(cost_value_loss)

    # total_loss += penalty_loss  # Do not add it to total loss!
    total_loss += policy._mean_cost_value_loss

    # Store stats in policy for stats_fn.
    policy._total_loss = total_loss
    policy._mean_policy_loss = mean_policy_loss
    policy._mean_vf_loss = mean_vf_loss
    policy._mean_entropy = mean_entropy
    policy._mean_kl = mean_kl

    return total_loss


def new_stats(policy, batch):
    ret = kl_and_loss_stats(policy, batch)
    ret["penalty"] = policy._penalty
    ret["penalty_param"] = policy._penalty_param
    ret["cost_loss"] = policy._mean_cost_loss
    ret["cost_value_loss"] = policy._mean_cost_value_loss
    return ret


def gradient_fn(policy, optimizer, loss):
    variables = policy.model.trainable_variables()

    assert len([v for v in variables if "penalty_param" in v.name]) == 0
    grads_and_vars = optimizer.compute_gradients(loss, variables)

    # Clip by global norm, if necessary.
    if policy.config["grad_clip"] is not None:
        grads = [g for (g, v) in grads_and_vars]
        policy.grads, _ = tf.clip_by_global_norm(grads, policy.config["grad_clip"])
        clipped_grads_and_vars = list(zip(policy.grads, variables))
        return clipped_grads_and_vars
    else:
        return grads_and_vars


class CentralizedCostAdvantage:
    """Following https://github.com/openai/safety-starter-agents/blob/master/safe_rl/pg/buffer.py#L72"""

    def __call__(self, samples):
        wrapped = False
        if isinstance(samples, SampleBatch):
            samples = MultiAgentBatch({DEFAULT_POLICY_ID: samples}, samples.count)
            wrapped = True

        for policy_id, batch in samples.policy_batches.items():
            cost_adv_mean = batch[COST_ADVANTAGE].mean()
            batch[COST_ADVANTAGE] -= cost_adv_mean

        if wrapped:
            samples = samples.policy_batches[DEFAULT_POLICY_ID]

        return samples


class UpdatePenalty:
    def __init__(self, workers):
        self.workers = workers

    def __call__(self, batch):
        def update(pi, pi_id):
            res = pi.update_penalty(batch)
            return (pi_id, res)

        res = self.workers.local_worker().foreach_trainable_policy(update)

        metrics = _get_shared_metrics()
        metrics.info["penalty_loss"] = res[0][1]

        return batch  # , fetch


def execution_plan(workers: WorkerSet, config: TrainerConfigDict) -> LocalIterator[dict]:
    """Execution plan of the PPO algorithm. Defines the distributed dataflow.

    Args:
        workers (WorkerSet): The WorkerSet for training the Polic(y/ies)
            of the Trainer.
        config (TrainerConfigDict): The trainer's configuration dict.

    Returns:
        LocalIterator[dict]: The Policy class to use with PPOTrainer.
            If None, use `default_policy` provided in build_trainer().
    """
    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # Collect batches for the trainable policies.
    rollouts = rollouts.for_each(
        SelectExperiences(workers.trainable_policies()))
    # Concatenate the SampleBatches into one.
    rollouts = rollouts.combine(
        ConcatBatches(min_batch_size=config["train_batch_size"]))
    # Standardize advantages.
    # <<<<< We add the cost advantage to normalization too! >>>>>
    rollouts = rollouts.for_each(StandardizeFields(["advantages", COST_ADVANTAGE]))

    # Update penalty
    rollouts = rollouts.for_each(UpdatePenalty(workers))

    # Perform one training step on the combined + standardized batch.
    if config["simple_optimizer"]:
        train_op = rollouts.for_each(
            TrainOneStep(
                workers,
                num_sgd_iter=config["num_sgd_iter"],
                sgd_minibatch_size=config["sgd_minibatch_size"]))
    else:
        train_op = rollouts.for_each(
            TrainTFMultiGPU(
                workers,
                sgd_minibatch_size=config["sgd_minibatch_size"],
                num_sgd_iter=config["num_sgd_iter"],
                num_gpus=config["num_gpus"],
                rollout_fragment_length=config["rollout_fragment_length"],
                num_envs_per_worker=config["num_envs_per_worker"],
                train_batch_size=config["train_batch_size"],
                shuffle_sequences=config["shuffle_sequences"],
                _fake_gpus=config["_fake_gpus"],
                framework=config.get("framework")))

    # Update KL after each round of training.
    train_op = train_op.for_each(lambda t: t[1]).for_each(UpdateKL(workers))

    # Warn about bad reward scales and return training metrics.
    return StandardMetricsReporting(train_op, workers, config) \
        .for_each(lambda result: warn_about_bad_reward_scales(config, result))


def make_model(policy, obs_space, action_space, config):
    dist_class, logit_dim = ModelCatalog.get_action_dist(action_space, config["model"])
    return ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=logit_dim,
        model_config=config["model"],
        framework="tf",
        model_interface=CostValueNetwork
    )


def setup_mixins_ppo_lag(policy, obs_space, action_space, config):
    setup_mixins(policy, obs_space, action_space, config)
    CostValueNetworkMixin.__init__(policy, obs_space, action_space, config)


def vf_preds_fetches(policy: Policy) -> Dict[str, TensorType]:
    return {
        SampleBatch.VF_PREDS: policy.model.value_function(),
        COST_VALUES: policy.model.get_cost_value(),
    }


def validate_config(config):
    original_validate_config(config)
    assert config["batch_mode"] == "complete_episodes", "We need to compute episode cost!"


class UpdatePenaltyMixin:

    def __init__(self):
        if hasattr(self, "_penalty_param") and self.config["worker_index"] == 0:
            ep_cost = tf.placeholder(
                tf.float32,
                shape=[None, ],
                name="ep_cost")

            with tf.control_dependencies([tf.print("Cost: ", ep_cost, " Param: ", self._penalty_param, self._penalty)]):
                penalty_loss = -self._penalty_param * (tf.reduce_mean(ep_cost) - self.config[COST_LIMIT])

            self._ep_cost_ph = ep_cost
            self._penalty_loss = penalty_loss

            self._penalty_optimizer = tf.train.AdamOptimizer(learning_rate=self.config[PENALTY_LR])
            self._train_penalty_op = self._penalty_optimizer.minimize(
                self._penalty_loss,
                var_list=[self._penalty_param],
                name="penalty_loss"
            )

    def update_penalty(self, batch):
        feed_dict = {
            self._is_training: True,
            self._ep_cost_ph: batch[COST].sum().reshape(-1, ) / batch[SampleBatch.DONES].sum(),
        }
        _, penalty_loss = self._sess.run([self._train_penalty_op, self._penalty_loss], feed_dict)
        return penalty_loss


def after_init(policy, obs_space, action_space, config):
    UpdatePenaltyMixin.__init__(policy)


PPOLagPolicy = PPOTFPolicy.with_updates(
    name="PPOLagPolicy",
    get_default_config=lambda: PPO_LAG_CONFIG,
    postprocess_fn=post_process_fn,
    loss_fn=ppo_lag_surrogate_loss,
    gradients_fn=gradient_fn,
    stats_fn=new_stats,
    make_model=make_model,
    before_loss_init=setup_mixins_ppo_lag,
    extra_action_fetches_fn=vf_preds_fetches,
    after_init=after_init,
    mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin, CostValueNetworkMixin,
            UpdatePenaltyMixin]
)

PPOLag = PPOTrainer.with_updates(
    name="PPOLag",
    default_policy=PPOLagPolicy,
    get_policy_class=lambda _: PPOLagPolicy,
    default_config=PPO_LAG_CONFIG,
    execution_plan=execution_plan,
    validate_config=validate_config
)
