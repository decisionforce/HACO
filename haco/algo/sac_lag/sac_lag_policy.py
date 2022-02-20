import logging
from ray.rllib.models.modelv2 import restore_original_dimensions
# from pgdrive.scene_creator.vehicle_module import PIDController
from collections import deque

import numpy as np
from gym.spaces import Box, Discrete
from haco.algo.sac_lag.sac_lag_model import ConstrainedSACModel
from ray.rllib.agents.ddpg.ddpg_tf_policy import ComputeTDErrorMixin, \
    TargetNetworkMixin
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio
from ray.rllib.execution.common import _get_shared_metrics
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import Beta, Categorical, \
    DiagGaussian, SquashedGaussian
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import try_import_tf, \
    try_import_tfp

tf, _, _ = try_import_tf()
tf1 = tf
tfp = try_import_tfp()

logger = logging.getLogger(__name__)

COST = "cost"
TOTAL_COST = "total_cost"

from ray.rllib.agents.sac.sac import DEFAULT_CONFIG
from ray.tune.utils.util import merge_dicts

SACPIDConfig = merge_dicts(DEFAULT_CONFIG,
                           {
                               "cost_limit": 3,
                               "recent_episode_num": 3,
                               "twin_cost_q": True,
                               "only_evaluate_cost": False,
                               "negative_cost_loss": False,  # useless now, only for runing old ckptcompatibility
                               "normalize": True,
                               "k_i": 0.01,
                               "k_p": 0.1,
                               "k_d": 0.0,
                               "info_cost_key": COST,
                               "info_total_cost_key": TOTAL_COST,
                           })


class PIDController:
    def __init__(self, k_p: float, k_i: float, k_d: float):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.p_error = 0
        self.i_error = 0
        self.d_error = 0

    def _update_error(self, current_error: float):
        self.i_error += current_error
        self.d_error = current_error - self.p_error
        self.p_error = current_error

    def get_result(self, current_error: float, make_up_coefficient=1.0):
        self._update_error(current_error)
        return (-self.k_p * self.p_error - self.k_i * self.i_error - self.k_d * self.d_error) * make_up_coefficient

    def reset(self):
        self.p_error = 0
        self.i_error = 0
        self.d_error = 0


def build_sac_model(policy, obs_space, action_space, config):
    # 2 cases:
    # 1) with separate state-preprocessor (before obs+action concat).
    # 2) no separate state-preprocessor: concat obs+actions right away.
    if config["use_state_preprocessor"]:
        num_outputs = 256  # Flatten last Conv2D to this many nodes.
    else:
        num_outputs = 0
        # No state preprocessor: fcnet_hiddens should be empty.
        if config["model"]["fcnet_hiddens"]:
            logger.warning(
                "When not using a state-preprocessor with SAC, `fcnet_hiddens`"
                " will be set to an empty list! Any hidden layer sizes are "
                "defined via `policy_model.fcnet_hiddens` and "
                "`Q_model.fcnet_hiddens`.")
            config["model"]["fcnet_hiddens"] = []

    # Force-ignore any additionally provided hidden layer sizes.
    # Everything should be configured using SAC's "Q_model" and "policy_model"
    # settings.
    policy.model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        model_interface=ConstrainedSACModel,
        name="sac_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"],
        twin_q=config["twin_q"],
        twin_cost_q=config["twin_cost_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"])

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        model_interface=ConstrainedSACModel,
        name="target_sac_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"],
        twin_q=config["twin_q"],
        twin_cost_q=config["twin_cost_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"])

    return policy.model


def postprocess_trajectory(policy,
                           sample_batch,
                           other_agent_batches=None,
                           episode=None):
    # if sample_batch.count > 1:
    #     raise ValueError
    # Put the actions to batch
    infos = sample_batch.get(SampleBatch.INFOS)
    if (infos is not None) and (infos[0] == 0.0):
        sample_batch[SampleBatch.INFOS] += 0.0
    if (infos is not None) and (infos[0] != 0.0):
        if "raw_action" in infos[0]:
            sample_batch[SampleBatch.ACTIONS] = np.array([info["raw_action"] for info in infos])
        sample_batch[policy.config["info_cost_key"]] = np.array(
            [info[policy.config["info_cost_key"]] for info in sample_batch[SampleBatch.INFOS]]
        ).astype(sample_batch[SampleBatch.REWARDS].dtype)
        sample_batch[policy.config["info_total_cost_key"]] = np.array(
            [info[policy.config["info_total_cost_key"]] for info in sample_batch[SampleBatch.INFOS]]
        ).astype(sample_batch[SampleBatch.REWARDS].dtype)
    else:
        assert episode is None, "Only during initialization, can we see empty infos."
        sample_batch[policy.config["info_cost_key"]] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
        sample_batch[policy.config["info_total_cost_key"]] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
    batch = postprocess_nstep_and_prio(policy, sample_batch)
    assert policy.config["info_cost_key"] in batch
    assert policy.config["info_total_cost_key"] in batch
    return batch


def get_dist_class(config, action_space):
    if isinstance(action_space, Discrete):
        raise ValueError()
        return Categorical
    else:
        if config["normalize_actions"]:
            return SquashedGaussian if \
                not config["_use_beta_distribution"] else Beta
        else:
            raise ValueError()
            return DiagGaussian


def get_distribution_inputs_and_class(policy,
                                      model,
                                      obs_batch,
                                      *,
                                      explore=True,
                                      **kwargs):
    # Get base-model output.
    model_out, state_out = model({
            "obs": obs_batch,
            "is_training": policy._get_is_training_placeholder(),
        }, [], None)
    # Get action model output from base-model output.
    if policy.config["image_obs"]:
        model_out = restore_original_dimensions(obs_batch,model.obs_space)
    distribution_inputs = model.get_policy_output(model_out)
    action_dist_class = get_dist_class(policy.config, policy.action_space)
    return distribution_inputs, action_dist_class, state_out


def sac_actor_critic_loss(policy, model, _, train_batch):
    _ = train_batch[policy.config["info_total_cost_key"]]  # Touch this item, this is helpful in ray 1.2.0

    # Setup the lambda multiplier.
    with tf.variable_scope('lambda'):
        param_init = 1e-8
        lambda_param = tf.get_variable(
            'lambda_value',
            initializer=float(param_init),
            trainable=False,
            dtype=tf.float32
        )
    policy.lambda_value = lambda_param

    # Should be True only for debugging purposes (e.g. test cases)!
    deterministic = policy.config["_deterministic_loss"]

    model_out_t, _ = model({
        "obs": train_batch[SampleBatch.CUR_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

    model_out_tp1, _ = model({
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

    target_model_out_tp1, _ = policy.target_model({
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

    # Discrete case.
    if model.discrete:
        raise ValueError("Doesn't support yet")
        # Get all action probs directly from pi and form their logp.
        log_pis_t = tf.nn.log_softmax(model.get_policy_output(model_out_t), -1)
        policy_t = tf.math.exp(log_pis_t)
        log_pis_tp1 = tf.nn.log_softmax(
            model.get_policy_output(model_out_tp1), -1)
        policy_tp1 = tf.math.exp(log_pis_tp1)
        # Q-values.
        q_t = model.get_q_values(model_out_t)
        # Target Q-values.
        q_tp1 = policy.target_model.get_q_values(target_model_out_tp1)
        if policy.config["twin_q"]:
            twin_q_t = model.get_twin_q_values(model_out_t)
            twin_q_tp1 = policy.target_model.get_twin_q_values(
                target_model_out_tp1)
            q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)
        q_tp1 -= model.alpha * log_pis_tp1

        # Actually selected Q-values (from the actions batch).
        one_hot = tf.one_hot(
            train_batch[SampleBatch.ACTIONS], depth=q_t.shape.as_list()[-1])
        q_t_selected = tf.reduce_sum(q_t * one_hot, axis=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = tf.reduce_sum(twin_q_t * one_hot, axis=-1)
        # Discrete case: "Best" means weighted by the policy (prob) outputs.
        q_tp1_best = tf.reduce_sum(tf.multiply(policy_tp1, q_tp1), axis=-1)
        q_tp1_best_masked = \
            (1.0 - tf.cast(train_batch[SampleBatch.DONES], tf.float32)) * \
            q_tp1_best
    # Continuous actions case.
    else:
        # Sample simgle actions from distribution.
        action_dist_class = get_dist_class(policy.config, policy.action_space)
        action_dist_t = action_dist_class(
            model.get_policy_output(model_out_t), policy.model)
        policy_t = action_dist_t.sample() if not deterministic else \
            action_dist_t.deterministic_sample()
        log_pis_t = tf.expand_dims(action_dist_t.logp(policy_t), -1)
        action_dist_tp1 = action_dist_class(
            model.get_policy_output(model_out_tp1), policy.model)
        policy_tp1 = action_dist_tp1.sample() if not deterministic else \
            action_dist_tp1.deterministic_sample()
        log_pis_tp1 = tf.expand_dims(action_dist_tp1.logp(policy_tp1), -1)

        # Q-values for the actually selected actions.
        q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
        if policy.config["twin_q"]:
            twin_q_t = model.get_twin_q_values(
                model_out_t, train_batch[SampleBatch.ACTIONS])

        # Cost Q-Value for actually selected actions
        c_q_t = model.get_cost_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
        if policy.config["twin_cost_q"]:
            twin_c_q_t = model.get_twin_cost_q_values(
                model_out_t, train_batch[SampleBatch.ACTIONS])

        # Q-values for current policy in given current state.
        q_t_det_policy = model.get_q_values(model_out_t, policy_t)
        if policy.config["twin_q"]:
            twin_q_t_det_policy = model.get_twin_q_values(
                model_out_t, policy_t)
            q_t_det_policy = tf.reduce_min(
                (q_t_det_policy, twin_q_t_det_policy), axis=0)

        # Cost Q-values for current policy in given current state.
        c_q_t_det_policy = model.get_cost_q_values(model_out_t, policy_t)
        if policy.config["twin_cost_q"]:
            twin_c_q_t_det_policy = model.get_twin_cost_q_values(
                model_out_t, policy_t)
            c_q_t_det_policy = tf.reduce_min(
                (c_q_t_det_policy, twin_c_q_t_det_policy), axis=0)

        # target q network evaluation
        q_tp1 = policy.target_model.get_q_values(target_model_out_tp1,
                                                 policy_tp1)
        if policy.config["twin_q"]:
            twin_q_tp1 = policy.target_model.get_twin_q_values(
                target_model_out_tp1, policy_tp1)
            # Take min over both twin-NNs.
            q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)

        # target c-q network evaluation
        c_q_tp1 = policy.target_model.get_cost_q_values(target_model_out_tp1,
                                                        policy_tp1)
        if policy.config["twin_cost_q"]:
            twin_c_q_tp1 = policy.target_model.get_twin_cost_q_values(
                target_model_out_tp1, policy_tp1)
            # Take min over both twin-NNs.
            c_q_tp1 = tf.reduce_min((c_q_tp1, twin_c_q_tp1), axis=0)

        q_t_selected = tf.squeeze(q_t, axis=len(q_t.shape) - 1)
        if policy.config["twin_q"]:
            twin_q_t_selected = tf.squeeze(twin_q_t, axis=len(twin_q_t.shape) - 1)

        # c_q_t selected
        c_q_t_selected = tf.squeeze(c_q_t, axis=len(c_q_t.shape) - 1)
        if policy.config["twin_cost_q"]:
            twin_c_q_t_selected = tf.squeeze(twin_c_q_t, axis=len(twin_c_q_t.shape) - 1)

        q_tp1 -= model.alpha * log_pis_tp1

        q_tp1_best = tf.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
        q_tp1_best_masked = (1.0 - tf.cast(train_batch[SampleBatch.DONES],
                                           tf.float32)) * q_tp1_best

    c_q_tp1_best = tf.squeeze(input=c_q_tp1, axis=len(c_q_tp1.shape) - 1)
    c_q_tp1_best_masked = \
        (1.0 - tf.cast(train_batch[SampleBatch.DONES], tf.float32)) * \
        c_q_tp1_best

    # compute RHS of bellman equation
    q_t_selected_target = tf.stop_gradient(
        train_batch[SampleBatch.REWARDS] +
        policy.config["gamma"] ** policy.config["n_step"] * q_tp1_best_masked)

    # Compute Cost of bellman equation.
    c_q_t_selected_target = tf.stop_gradient(train_batch[policy.config["info_cost_key"]] +
                                             policy.config["gamma"] ** policy.config["n_step"] * c_q_tp1_best_masked)

    # Compute the TD-error (potentially clipped).
    base_td_error = tf.math.abs(q_t_selected - q_t_selected_target)
    if policy.config["twin_q"]:
        twin_td_error = tf.math.abs(twin_q_t_selected - q_t_selected_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error

    # Compute the Cost TD-error (potentially clipped).
    base_c_td_error = tf.math.abs(c_q_t_selected - c_q_t_selected_target)
    if policy.config["twin_cost_q"]:
        twin_c_td_error = tf.math.abs(twin_c_q_t_selected - c_q_t_selected_target)
        c_td_error = 0.5 * (base_c_td_error + twin_c_td_error)
    else:
        c_td_error = base_c_td_error

    critic_loss = [
        0.5 * tf.keras.losses.MSE(
            y_true=q_t_selected_target, y_pred=q_t_selected)
    ]
    if policy.config["twin_q"]:
        critic_loss.append(0.5 * tf.keras.losses.MSE(
            y_true=q_t_selected_target, y_pred=twin_q_t_selected))

    # add cost critic
    critic_loss.append(
        0.5 * tf.keras.losses.MSE(
            y_true=c_q_t_selected_target, y_pred=c_q_t_selected))
    if policy.config["twin_cost_q"]:
        critic_loss.append(0.5 * tf.keras.losses.MSE(
            y_true=c_q_t_selected_target, y_pred=twin_c_q_t_selected))

    # Alpha- and actor losses.
    # Note: In the papers, alpha is used directly, here we take the log.
    # Discrete case: Multiply the action probs as weights with the original
    # loss terms (no expectations needed).
    if model.discrete:
        raise ValueError("Didn't support discrete mode yet")
        alpha_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.multiply(
                    tf.stop_gradient(policy_t), -model.log_alpha *
                                                tf.stop_gradient(log_pis_t + model.target_entropy)),
                axis=-1))
        actor_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.multiply(
                    # NOTE: No stop_grad around policy output here
                    # (compare with q_t_det_policy for continuous case).
                    policy_t,
                    model.alpha * log_pis_t - tf.stop_gradient(q_t)),
                axis=-1))
    else:
        alpha_loss = -tf.reduce_mean(
            model.log_alpha *
            tf.stop_gradient(log_pis_t + model.target_entropy))
        if policy.config["only_evaluate_cost"]:
            actor_loss = tf.reduce_mean(
                model.alpha * log_pis_t - q_t_det_policy)
            cost_loss = 0
            reward_loss = actor_loss
        else:
            reward_loss = tf.reduce_mean(
                model.alpha * log_pis_t - q_t_det_policy)
            cost_loss = tf.reduce_mean(policy.lambda_value * c_q_t_det_policy)
            actor_loss = tf.reduce_mean(
                model.alpha * log_pis_t - q_t_det_policy + policy.lambda_value * c_q_t_det_policy)
        actor_loss = actor_loss / (1 + policy.lambda_value) if policy.config["normalize"] else actor_loss

    # save for stats function
    policy.policy_t = policy_t
    policy.cost_loss = cost_loss
    policy.reward_loss = reward_loss
    policy.mean_batch_cost = train_batch[policy.config["info_cost_key"]]
    policy.q_t = q_t
    policy.c_q_tp1 = c_q_tp1
    policy.c_q_t = c_q_t
    policy.td_error = td_error
    policy.c_td_error = c_td_error
    policy.actor_loss = actor_loss
    policy.critic_loss = critic_loss
    policy.c_td_target = c_q_t_selected_target
    policy.alpha_loss = alpha_loss
    policy.alpha_value = model.alpha
    policy.target_entropy = model.target_entropy

    # in a custom apply op we handle the losses separately, but return them
    # combined in one loss for now
    return actor_loss + tf.math.add_n(critic_loss) + alpha_loss


def stats(policy, train_batch):
    return {
        # "policy_t": policy.policy_t,
        # "td_error": policy.td_error,
        "mean_td_error": tf.reduce_mean(policy.td_error),
        "mean_c_td_error": tf.reduce_mean(policy.c_td_error),
        "actor_loss": tf.reduce_mean(policy.actor_loss),
        "critic_loss": tf.reduce_mean(policy.critic_loss[:2] if policy.config["twin_q"] else policy.critic_loss[0]),
        "cost_critic_loss": tf.reduce_mean(
            policy.critic_loss[2:] if policy.config["twin_q"] else policy.critic_loss[1]),
        "alpha_loss": tf.reduce_mean(policy.alpha_loss),
        "lambda_value": tf.reduce_mean(policy.lambda_value),
        "alpha_value": tf.reduce_mean(policy.alpha_value),
        "target_entropy": tf.constant(policy.target_entropy),
        "c_td_target": tf.reduce_mean(policy.c_td_target),
        "mean_q": tf.reduce_mean(policy.q_t),
        "mean_c_q": tf.reduce_mean(policy.c_q_t),
        "max_q": tf.reduce_max(policy.q_t),
        "max_c_q": tf.reduce_max(policy.c_q_t),
        "min_q": tf.reduce_min(policy.q_t),
        "min_c_q": tf.reduce_min(policy.c_q_t),
        "c_q_tp1": tf.reduce_mean(policy.c_q_tp1),
        "mean_batch_cost": tf.reduce_mean(policy.mean_batch_cost),
        "reward_loss": tf.reduce_mean(policy.reward_loss),
        "cost_loss": tf.reduce_mean(policy.cost_loss)
    }


def setup_early_mixins(policy, obs_space, action_space, config):
    ActorCriticOptimizerMixin.__init__(policy, config)


def setup_mid_mixins(policy, obs_space, action_space, config):
    ComputeTDErrorMixin.__init__(policy, sac_actor_critic_loss)


def setup_late_mixins(policy, obs_space, action_space, config):
    TargetNetworkMixin.__init__(policy, config)
    UpdatePenaltyMixin.__init__(policy)


def validate_spaces(pid, observation_space, action_space, config):
    if not isinstance(action_space, (Box, Discrete)):
        raise UnsupportedSpaceException(
            "Action space ({}) of {} is not supported for "
            "SAC.".format(action_space, pid))
    if isinstance(action_space, Box) and len(action_space.shape) > 1:
        raise UnsupportedSpaceException(
            "Action space ({}) of {} has multiple dimensions "
            "{}. ".format(action_space, pid, action_space.shape) +
            "Consider reshaping this into a single dimension, "
            "using a Tuple action space, or the multi-agent API.")


class UpdatePenalty:
    def __init__(self, workers):
        self.workers = workers

    def __call__(self, batch):
        def update(pi, pi_id):
            res = pi.update_penalty(batch)
            return (pi_id, res)

        res = self.workers.local_worker().foreach_trainable_policy(update)

        metrics = _get_shared_metrics()
        metrics.info["pid_error"] = res[0][1][0]
        metrics.info["mean_online_episode_cost"] = res[0][1][1]

        return batch  # , fetch


# from ppo-lag
class UpdatePenaltyMixin:

    def __init__(self):
        if hasattr(self, "lambda_value") and self.config["worker_index"] == 0:
            self.recent_episode_cost = deque(maxlen=self.config["recent_episode_num"])
            self.pid_controller = PIDController(self.config["k_p"], self.config["k_i"], self.config["k_d"])
            self.new_error = 0
            self.online_cost = 0

    def update_penalty(self, batch: SampleBatch):
        assert self.config["info_total_cost_key"] in batch
        # if batch.get(SampleBatch.INFOS) is None:
        #     return 0, 0
        for i in range(batch.count):
            if batch[SampleBatch.DONES][i]:
                self.recent_episode_cost.append(batch[self.config["info_total_cost_key"]][i])
                self.online_cost = mean_episode_cost = np.array(
                    [np.sum(self.recent_episode_cost) / len(self.recent_episode_cost)])
                self.new_error = mean_episode_cost - self.config["cost_limit"]
                # new_lambda = np.exp(-self.pid_controller.get_result(self.new_error))[0]

                pid_result = self.pid_controller.get_result(self.new_error)
                pid_result = np.clip(pid_result, -300, 300)
                # print("PIDRESULT: {}, Error {}, P {}, I {}, D {}".format(pid_result,
                #         self.new_error,
                #         self.pid_controller.p_error,
                #         self.pid_controller.i_error,
                #         self.pid_controller.d_error,
                #     ))

                new_lambda = np.log(np.exp(-pid_result)[0] + 1)
                assign_op = self.lambda_value.assign(new_lambda)
                self._sess.run(assign_op)
        return self.new_error, self.online_cost


def gradients_fn(policy, optimizer, loss):
    # Eager: Use GradientTape.
    if policy.config["framework"] in ["tf2", "tfe"]:
        raise ValueError()
    # Tf1.x: Use optimizer.compute_gradients()
    else:
        actor_grads_and_vars = policy._actor_optimizer.compute_gradients(
            policy.actor_loss, var_list=policy.model.policy_variables())

        q_weights = policy.model.q_variables()
        c_q_weights = policy.model.cost_q_variables()
        if policy.config["twin_q"]:
            half_cutoff = len(q_weights) // 2
            base_q_optimizer, twin_q_optimizer = policy._critic_optimizer[0:2]
            critic_grads_and_vars = base_q_optimizer.compute_gradients(
                policy.critic_loss[0], var_list=q_weights[:half_cutoff]
            ) + twin_q_optimizer.compute_gradients(
                policy.critic_loss[1], var_list=q_weights[half_cutoff:])
        else:
            critic_grads_and_vars = policy._critic_optimizer[
                0].compute_gradients(
                policy.critic_loss[0], var_list=q_weights)
        if policy.config["twin_cost_q"]:
            c_half_cutoff = len(c_q_weights) // 2
            base_c_q_optimizer, twin_c_q_optimizer = policy._critic_optimizer[-2:]
            c_critic_grads_and_vars = base_c_q_optimizer.compute_gradients(
                policy.critic_loss[-2], var_list=c_q_weights[:c_half_cutoff]
            ) + twin_q_optimizer.compute_gradients(
                policy.critic_loss[-1], var_list=c_q_weights[c_half_cutoff:])
        else:
            c_critic_grads_and_vars = policy._critic_optimizer[
                -1].compute_gradients(
                policy.critic_loss[-1], var_list=c_q_weights)
        alpha_grads_and_vars = policy._alpha_optimizer.compute_gradients(
            policy.alpha_loss, var_list=[policy.model.log_alpha])

    # Clip if necessary.
    if policy.config["grad_clip"]:
        clip_func = tf.clip_by_norm
    else:
        clip_func = tf.identity

    # Save grads and vars for later use in `build_apply_op`.
    policy._actor_grads_and_vars = [(clip_func(g), v)
                                    for (g, v) in actor_grads_and_vars
                                    if g is not None]
    policy._critic_grads_and_vars = [(clip_func(g), v)
                                     for (g, v) in critic_grads_and_vars
                                     if g is not None]
    # for cost critic
    policy._c_critic_grads_and_vars = [(clip_func(g), v)
                                       for (g, v) in c_critic_grads_and_vars
                                       if g is not None]

    policy._alpha_grads_and_vars = [(clip_func(g), v)
                                    for (g, v) in alpha_grads_and_vars
                                    if g is not None]

    grads_and_vars = (
            policy._actor_grads_and_vars + policy._critic_grads_and_vars + policy._c_critic_grads_and_vars +
            policy._alpha_grads_and_vars)
    return grads_and_vars


def apply_gradients(policy, optimizer, grads_and_vars):
    actor_apply_ops = policy._actor_optimizer.apply_gradients(policy._actor_grads_and_vars)

    cgrads = policy._critic_grads_and_vars
    c_cgrads = policy._c_critic_grads_and_vars
    if policy.config["twin_q"]:
        half_cutoff = len(cgrads) // 2
        critic_apply_ops = [
            policy._critic_optimizer[0].apply_gradients(cgrads[:half_cutoff]),
            policy._critic_optimizer[1].apply_gradients(cgrads[half_cutoff:])]

    else:
        critic_apply_ops = [
            policy._critic_optimizer[0].apply_gradients(cgrads)]
    if policy.config["twin_cost_q"]:
        c_half_cutoff = len(c_cgrads) // 2
        critic_apply_ops += [policy._critic_optimizer[-2].apply_gradients(c_cgrads[:c_half_cutoff]),
                             policy._critic_optimizer[-1].apply_gradients(c_cgrads[c_half_cutoff:])]
    else:
        critic_apply_ops.append(policy._critic_optimizer[-1].apply_gradients(c_cgrads))

    if policy.config["framework"] in ["tf2", "tfe"]:
        policy._alpha_optimizer.apply_gradients(policy._alpha_grads_and_vars)
        return
    else:
        alpha_apply_ops = policy._alpha_optimizer.apply_gradients(
            policy._alpha_grads_and_vars,
            global_step=tf1.train.get_or_create_global_step())

        return tf.group([actor_apply_ops, alpha_apply_ops] + critic_apply_ops)


class ActorCriticOptimizerMixin:
    def __init__(self, config):
        # - Create global step for counting the number of update operations.
        # - Use separate optimizers for actor & critic.
        if config["framework"] in ["tf2", "tfe"]:
            raise ValueError
            self.global_step = get_variable(0, tf_name="global_step")
            self._actor_optimizer = tf.keras.optimizers.Adam(
                learning_rate=config["optimization"]["actor_learning_rate"])
            self._critic_optimizer = [
                tf.keras.optimizers.Adam(learning_rate=config["optimization"][
                    "critic_learning_rate"])
            ]
            if config["twin_q"]:
                self._critic_optimizer.append(
                    tf.keras.optimizers.Adam(learning_rate=config[
                        "optimization"]["critic_learning_rate"]))
            self._alpha_optimizer = tf.keras.optimizers.Adam(
                learning_rate=config["optimization"]["entropy_learning_rate"])

            # add optimizer for cost value
            self._critic_optimizer.append(
                tf.keras.optimizers.Adam(learning_rate=config[
                    "optimization"]["critic_learning_rate"]))
            if config["twin_cost_q"]:
                self._critic_optimizer.append(
                    tf.keras.optimizers.Adam(learning_rate=config[
                        "optimization"]["critic_learning_rate"]))
        else:
            self.global_step = tf1.train.get_or_create_global_step()
            self._actor_optimizer = tf1.train.AdamOptimizer(
                learning_rate=config["optimization"]["actor_learning_rate"])
            self._critic_optimizer = [
                tf1.train.AdamOptimizer(learning_rate=config["optimization"][
                    "critic_learning_rate"])
            ]
            if config["twin_q"]:
                self._critic_optimizer.append(
                    tf1.train.AdamOptimizer(learning_rate=config[
                        "optimization"]["critic_learning_rate"]))
            self._alpha_optimizer = tf1.train.AdamOptimizer(
                learning_rate=config["optimization"]["entropy_learning_rate"])

            # add optimizer for cost value
            self._critic_optimizer.append(
                tf1.train.AdamOptimizer(learning_rate=config[
                    "optimization"]["critic_learning_rate"]))
            if config["twin_cost_q"]:
                self._critic_optimizer.append(
                    tf1.train.AdamOptimizer(learning_rate=config[
                        "optimization"]["critic_learning_rate"]))


SACPIDPolicy = build_tf_policy(
    name="SACPIDPolicy",
    get_default_config=lambda: SACPIDConfig,
    make_model=build_sac_model,
    postprocess_fn=postprocess_trajectory,
    action_distribution_fn=get_distribution_inputs_and_class,
    loss_fn=sac_actor_critic_loss,
    stats_fn=stats,
    gradients_fn=gradients_fn,
    apply_gradients_fn=apply_gradients,
    extra_learn_fetches_fn=lambda policy: {"td_error": policy.td_error},
    mixins=[
        TargetNetworkMixin, ActorCriticOptimizerMixin, ComputeTDErrorMixin, UpdatePenaltyMixin
    ],
    validate_spaces=validate_spaces,
    before_init=setup_early_mixins,
    before_loss_init=setup_mid_mixins,
    after_init=setup_late_mixins,
    obs_include_prev_action_reward=False)
