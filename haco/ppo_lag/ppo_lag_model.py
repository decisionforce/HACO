import numpy as np
from ray.rllib.agents.ppo.ppo_tf_policy import SampleBatch
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.framework import get_activation_fn
from ray.rllib.utils.tf_ops import make_tf_callable

tf, _, _ = try_import_tf()


class CostValueNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        if config.get("use_gae"):

            @make_tf_callable(self.get_session())
            def cost_value(ob, prev_action, prev_reward, *state):
                model_out, _ = self.model(
                    {
                        SampleBatch.CUR_OBS: tf.convert_to_tensor([ob]),
                        SampleBatch.PREV_ACTIONS: tf.convert_to_tensor([prev_action]),
                        SampleBatch.PREV_REWARDS: tf.convert_to_tensor([prev_reward]),
                        "is_training": tf.convert_to_tensor(False),
                    }, [tf.convert_to_tensor([s]) for s in state], tf.convert_to_tensor([1])
                )
                return self.model.get_cost_value()[0]
        else:

            @make_tf_callable(self.get_session())
            def cost_value(ob, prev_action, prev_reward, *state):
                return tf.constant(0.0)

        self._cost_value = cost_value


class CostValueNetwork(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CostValueNetwork, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        activation = get_activation_fn(model_config.get("fcnet_activation"))
        hiddens = model_config.get("fcnet_hiddens")
        # we are using obs_flat, so take the flattened shape as input
        inputs = tf.keras.layers.Input(shape=(np.product(obs_space.shape),), name="observations")
        # build the value network for cost
        last_layer = inputs
        i = 1
        for size in hiddens:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_value_cost_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer(1.0)
            )(last_layer)
            i += 1
        value_out_cost = tf.keras.layers.Dense(
            1, name="value_out_cost", activation=None, kernel_initializer=normc_initializer(0.01)
        )(last_layer)

        # Register the network
        self.cost_value_network = tf.keras.Model(inputs, value_out_cost)
        self.register_variables(self.cost_value_network.variables)
        self._last_cost_value = None

    def forward(self, input_dict, state, seq_lens):
        ret = super(CostValueNetwork, self).forward(input_dict, state, seq_lens)
        self._last_cost_value = self.cost_value_network(input_dict["obs_flat"])
        return ret

    def get_cost_value(self):
        return tf.reshape(self._last_cost_value, [-1])
