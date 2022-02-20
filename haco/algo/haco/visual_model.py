import numpy as np
from gym.spaces import Discrete
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf

tf, _, _ = try_import_tf()

SCALE_DIAG_MIN_MAX = (-20, 2)


class VisualConstrainedSACModel(TFModelV2):
    """Extension of standard TFModel for SAC.

    Data flow:
        obs -> forward() -> model_out
        model_out -> get_policy_output() -> pi(s)
        model_out, actions -> get_q_values() -> Q(s, a)
        model_out, actions -> get_twin_q_values() -> Q_twin(s, a)

    Note that this class by itself is not a valid model unless you
    implement forward() in a subclass."""

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            actor_hidden_activation="relu",
            actor_hiddens=(256, 256),
            critic_hidden_activation="relu",
            critic_hiddens=(256, 256),
            twin_q=False,
            twin_cost_q=False,
            initial_alpha=1.0,
            target_entropy=None
    ):
        """Initialize variables of this model.

        Extra model kwargs:
            actor_hidden_activation (str): activation for actor network
            actor_hiddens (list): hidden layers sizes for actor network
            critic_hidden_activation (str): activation for critic network
            critic_hiddens (list): hidden layers sizes for critic network
            twin_q (bool): build twin Q networks.
            initial_alpha (float): The initial value for the to-be-optimized
                alpha parameter (default: 1.0).

        Note that the core layers for forward() are not defined here, this
        only defines the layers for the output heads. Those layers for
        forward() should be defined in subclasses of SACModel.
        """
        super(VisualConstrainedSACModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        self.discrete = False
        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
            self.discrete = True
            action_outs = q_outs = self.action_dim
        else:
            self.action_dim = np.product(action_space.shape)
            action_outs = 2 * self.action_dim
            q_outs = 1

        # build image model
        self.img_input = tf.keras.layers.Input(
            shape=(84, 84, 5), name="model_out"
        )
        self.conv_out = self.conv_nets(self.img_input, "relu", "feature_abstract")

        # concat speed
        self.model_out = tf.keras.layers.Input(
            shape=(257,), name="model_out"
        )

        self.action_model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=hidden,
                    activation=getattr(tf.nn, actor_hidden_activation, None),
                    name="action_{}".format(i + 1)
                ) for i, hidden in enumerate(actor_hiddens)
            ] + [
                tf.keras.layers.
            Dense(units=action_outs, activation=None, name="action_out")
            ]
        )
        self.shift_and_log_scale_diag = self.action_model(self.model_out)

        self.register_variables(self.action_model.variables)

        self.actions_input = None
        if not self.discrete:
            self.actions_input = tf.keras.layers.Input(
                shape=(self.action_dim,), name="actions"
            )

        def build_q_net(name, observations, actions):
            # For continuous actions: Feed obs and actions (concatenated)
            # through the NN. For discrete actions, only obs.
            q_net = tf.keras.Sequential(
                (
                    [
                        tf.keras.layers.Concatenate(axis=1),
                    ] if not self.discrete else []
                ) + [
                    tf.keras.layers.Dense(
                        units=units,
                        activation=getattr(
                            tf.nn, critic_hidden_activation, None
                        ),
                        kernel_initializer=normc_initializer(1.0),
                        name="{}_hidden_{}".format(name, i)
                    ) for i, units in enumerate(critic_hiddens)
                ] + [
                    tf.keras.layers.Dense(
                        units=q_outs,
                        activation=None,
                        kernel_initializer=normc_initializer(1.0),
                        name="{}_out".format(name)
                    )
                ]
            )

            if self.discrete:
                q_net = tf.keras.Model(observations, q_net(observations))
            else:
                q_net = tf.keras.Model(
                    [observations, actions], q_net([observations, actions])
                )
            return q_net

        self.q_net = build_q_net("q", self.model_out, self.actions_input)
        self.cost_q_net = build_q_net(
            "cost_q", self.model_out, self.actions_input
        )
        self.register_variables(self.q_net.variables)
        self.register_variables(self.cost_q_net.variables)

        if twin_q:
            self.twin_q_net = build_q_net(
                "twin_q", self.model_out, self.actions_input
            )
            self.register_variables(self.twin_q_net.variables)
        else:
            self.twin_q_net = None
        if twin_cost_q:
            self.cost_twin_q_net = build_q_net(
                "cost_twin_q", self.model_out, self.actions_input
            )
            self.register_variables(self.cost_twin_q_net.variables)
        else:
            self.cost_twin_q_net = None

        self.log_alpha = tf.Variable(
            np.log(initial_alpha), dtype=tf.float32, name="log_alpha"
        )
        self.alpha = tf.exp(self.log_alpha)

        # Auto-calculate the target entropy.
        if target_entropy is None or target_entropy == "auto":
            # See hyperparams in [2] (README.md).
            if self.discrete:
                target_entropy = 0.98 * np.array(
                    -np.log(1.0 / action_space.n), dtype=np.float32
                )
            # See [1] (README.md).
            else:
                target_entropy = -np.prod(action_space.shape)
        self.target_entropy = target_entropy

        self.register_variables([self.log_alpha])

    def get_q_values(self, model_out, actions=None):
        """Return the Q estimates for the most recent forward pass.

        This implements Q(s, a).

        Arguments:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].
            actions (Optional[Tensor]): Actions to return the Q-values for.
                Shape: [BATCH_SIZE, action_dim]. If None (discrete action
                case), return Q-values for all actions.

        Returns:
            tensor of shape [BATCH_SIZE].
        """
        model_out = self.get_model_out(model_out)
        if actions is not None:
            return self.q_net([model_out, actions])
        else:
            return self.q_net(model_out)

    def get_cost_q_values(self, model_out, actions=None):
        model_out = self.get_model_out(model_out)
        if actions is not None:
            return self.cost_q_net([model_out, actions])
        else:
            return self.cost_q_net(model_out)

    def get_twin_cost_q_values(self, model_out, actions=None):
        model_out = self.get_model_out(model_out)
        if actions is not None:
            return self.cost_twin_q_net([model_out, actions])
        else:
            return self.cost_twin_q_net(model_out)

    def get_twin_q_values(self, model_out, actions=None):
        """Same as get_q_values but using the twin Q net.

        This implements the twin Q(s, a).

        Arguments:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].
            actions (Optional[Tensor]): Actions to return the Q-values for.
                Shape: [BATCH_SIZE, action_dim]. If None (discrete action
                case), return Q-values for all actions.

        Returns:
            tensor of shape [BATCH_SIZE].
        """
        model_out = self.get_model_out(model_out)
        if actions is not None:
            return self.twin_q_net([model_out, actions])
        else:
            return self.twin_q_net(model_out)

    def get_policy_output(self, model_out):
        """Return the action output for the most recent forward pass.

        This outputs the support for pi(s). For continuous action spaces, this
        is the action directly. For discrete, is is the mean / std dev.

        Arguments:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].

        Returns:
            tensor of shape [BATCH_SIZE, action_out_size]
        """
        model_out = self.get_model_out(model_out)
        return self.action_model(model_out)

    def policy_variables(self):
        """Return the list of variables for the policy net."""

        return list(self.action_model.variables)

    def q_variables(self):
        """Return the list of variables for Q / twin Q nets."""

        return self.q_net.variables + self.conv_model.variables + (
            self.twin_q_net.variables if self.twin_q_net else []
        )

    def cost_q_variables(self):
        return self.cost_q_net.variables + self.conv_model.variables + (
            self.cost_twin_q_net.variables
            if self.cost_twin_q_net else []
        )

    def get_model_out(self, obs):
        bird_v = obs["birdview"]
        speed = obs["speed"]
        conv_out = self.conv_model(bird_v)
        return tf.concat([conv_out, speed], axis=1)

    def conv_nets(self, image_inputs, activation, name):
        # conv 1
        conv_model = tf.keras.Sequential([
            # tf.keras.layers.Conv2D(
            #     32, [5, 5],
            #     strides=[1, 1],
            #     activation=activation,
            #     padding="same",
            #     data_format="channels_last",
            #     name="{}_conv_0".format(name)
            # ),

            # conv 2
            tf.keras.layers.Conv2D(
                64, [3, 3],
                strides=[1, 1],
                activation=activation,
                padding="same",
                data_format="channels_last",
                name="{}_conv_3".format(name)
            ),

            # pooling
            tf.keras.layers.AveragePooling2D(
                pool_size=(4, 4), strides=None, padding="valid", data_format="channels_last",
                name="{}_pooling_1".format(name)
            ),

            # conv 3
            tf.keras.layers.Conv2D(
                128, [5, 5],
                strides=[2, 2],
                activation=activation,
                padding="valid",
                data_format="channels_last",
                name="{}_conv_5".format(name)
            ),

            # conv 4
            tf.keras.layers.Conv2D(
                256, [3, 3],
                strides=[2, 2],
                activation=activation,
                padding="valid",
                data_format="channels_last",
                name="{}_conv_7".format(name)
            ),

            # # pooling
            # tf.keras.layers.AveragePooling2D(
            #     pool_size=(4, 4), strides=None, padding="valid", data_format="channels_last",
            #     name="{}_pooling_2".format(name)
            # ),

            # flatten
            tf.keras.layers.Flatten(data_format="channels_last")])
        conv_net = conv_model(image_inputs)
        self.conv_model = conv_model
        self.register_variables(conv_model.variables)
        return conv_net
