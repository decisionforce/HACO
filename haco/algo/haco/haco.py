# from pgdrive.scene_creator.vehicle_module import PIDController
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf, \
    try_import_tfp

from haco.algo.haco.visual_model import VisualConstrainedSACModel
from haco.algo.sac_lag.sac_lag_model import ConstrainedSACModel

tf, _, _ = try_import_tf()
tf1 = tf
tfp = try_import_tfp()
import numpy as np
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune.utils.util import merge_dicts

from haco.algo.sac_lag.sac_lag import SACLagTrainer, validate_config
from haco.algo.sac_lag.sac_lag_policy import SACPIDConfig, SACPIDPolicy, get_dist_class

# Update penalty
#

NEWBIE_ACTION = "newbie_action"
TAKEOVER = "takeover"

HACOConfig = merge_dicts(SACPIDConfig,
                         {
                                    "info_cost_key": "takeover_cost",
                                    "info_total_cost_key": "total_takeover_cost",
                                    "takeover_data_discard": False,  # useless
                                    "alpha": 10.0,
                                    "no_reward": True,  # this will disable the native reward from env
                                    "image_obs": False

                                })


def validate_saver_config(config):
    validate_config(config)
    assert config["info_cost_key"] == "takeover_cost" and config["info_total_cost_key"] == "total_takeover_cost"


def postprocess_trajectory(policy,
                           sample_batch,
                           other_agent_batches=None,
                           episode=None):
    # if sample_batch.count > 1:
    #     raise ValueError
    # Put the actions to batch
    infos = sample_batch.get(SampleBatch.INFOS)
    if (infos is not None) and (infos[0] != 0.0):
        sample_batch[NEWBIE_ACTION] = sample_batch.copy()[SampleBatch.ACTIONS]
        sample_batch[SampleBatch.ACTIONS] = np.array([info["raw_action"] for info in infos])
        sample_batch[TAKEOVER] = np.array(
            [info[TAKEOVER] for info in sample_batch[SampleBatch.INFOS]])
        sample_batch[policy.config["info_cost_key"]] = np.array(
            [info[policy.config["info_cost_key"]] for info in sample_batch[SampleBatch.INFOS]]
        ).astype(sample_batch[SampleBatch.REWARDS].dtype)
        sample_batch[policy.config["info_total_cost_key"]] = np.array(
            [info[policy.config["info_total_cost_key"]] for info in sample_batch[SampleBatch.INFOS]]
        ).astype(sample_batch[SampleBatch.REWARDS].dtype)
        if policy.config["no_reward"]:
            sample_batch[SampleBatch.REWARDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
    else:
        assert episode is None, "Only during initialization, can we see empty infos."
        sample_batch[policy.config["info_cost_key"]] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
        sample_batch[policy.config["info_total_cost_key"]] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
        sample_batch[NEWBIE_ACTION] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
        sample_batch[TAKEOVER] = np.zeros_like(sample_batch[SampleBatch.DONES])
    batch = postprocess_nstep_and_prio(policy, sample_batch)
    assert policy.config["info_cost_key"] in batch
    assert policy.config["info_total_cost_key"] in batch
    assert TAKEOVER in batch
    assert NEWBIE_ACTION in batch
    return batch


def sac_actor_critic_loss(policy, model, _, train_batch):
    _ = train_batch[policy.config["info_total_cost_key"]]  # Touch this item, this is helpful in ray 1.2.0
    takeover_mask = (tf.cast(train_batch[TAKEOVER], tf.float32))
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
    if policy.config["image_obs"]:
        model_out_t = restore_original_dimensions(train_batch[SampleBatch.CUR_OBS], model.obs_space)

    model_out_tp1, _ = model({
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)
    if policy.config["image_obs"]:
        model_out_tp1 = restore_original_dimensions(train_batch[SampleBatch.NEXT_OBS], model.obs_space)

    target_model_out_tp1, _ = policy.target_model({
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)
    if policy.config["image_obs"]:
        target_model_out_tp1 = restore_original_dimensions(train_batch[SampleBatch.NEXT_OBS], model.obs_space)

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
    q_t = model.get_q_values(model_out_t,
                             train_batch[SampleBatch.ACTIONS])
    if policy.config["twin_q"]:
        twin_q_t = model.get_twin_q_values(
            model_out_t, train_batch[SampleBatch.ACTIONS])

    # Cost Q-Value for actually selected actions
    c_q_t = model.get_cost_q_values(model_out_t,
                                    train_batch[NEWBIE_ACTION])
    if policy.config["twin_cost_q"]:
        twin_c_q_t = model.get_twin_cost_q_values(
            model_out_t,
            train_batch[NEWBIE_ACTION])

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

    # conservative loss
    newbie_q_t = model.get_q_values(model_out_t, train_batch[NEWBIE_ACTION])
    if policy.config["twin_q"]:
        newbie_twin_q_t = model.get_twin_q_values(
            model_out_t, train_batch[NEWBIE_ACTION])

    newbie_q_t_selected = tf.squeeze(newbie_q_t, axis=len(newbie_q_t.shape) - 1)
    if policy.config["twin_q"]:
        newbie_twin_q_t_selected = tf.squeeze(newbie_twin_q_t, axis=len(newbie_twin_q_t.shape) - 1)

    # add conservative loss
    critic_loss = [
        0.5 * tf.keras.losses.MSE(
            y_true=q_t_selected_target, y_pred=q_t_selected) - tf.reduce_mean(
            takeover_mask * policy.config["alpha"] * (q_t_selected - newbie_q_t_selected))]
    if policy.config["twin_q"]:
        loss = 0.5 * tf.keras.losses.MSE(y_true=q_t_selected_target, y_pred=twin_q_t_selected) - tf.reduce_mean(
            takeover_mask *
            policy.config[
                "alpha"] * (
                    twin_q_t_selected - newbie_twin_q_t_selected))
        critic_loss.append(loss)

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
            c_q_t_det_policy -= -policy.config["cost_limit"]
            if policy.config["cost_limit"] != -1:
                c_q_t_det_policy *= policy.lambda_value
            cost_loss = tf.reduce_mean(c_q_t_det_policy)
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


def build_sac_model(policy, obs_space, action_space, config):
    # 2 cases:
    # 1) with separate state-preprocessor (before obs+action concat).
    # 2) no separate state-preprocessor: concat obs+actions right away.
    if config["use_state_preprocessor"]:
        num_outputs = 256  # Flatten last Conv2D to this many nodes.
    else:
        num_outputs = 0
        # No state preprocessor: fcnet_hiddens should be empty.

    # Force-ignore any additionally provided hidden layer sizes.
    # Everything should be configured using SAC's "Q_model" and "policy_model"
    # settings.
    policy.model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        model_interface=ConstrainedSACModel if not policy.config["image_obs"] else VisualConstrainedSACModel,
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
        model_interface=ConstrainedSACModel if not policy.config["image_obs"] else VisualConstrainedSACModel,
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


HACOPolicy = SACPIDPolicy.with_updates(name="HACO",
                                       make_model=build_sac_model,
                                       get_default_config=lambda: HACOConfig,
                                       postprocess_fn=postprocess_trajectory,
                                       loss_fn=sac_actor_critic_loss)

HACOTrainer = SACLagTrainer.with_updates(name="HACO",
                                         default_config=HACOConfig,
                                         default_policy=HACOPolicy,
                                         get_policy_class=lambda config: HACOPolicy,
                                         validate_config=validate_config,
                                         )
