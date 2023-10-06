from collections import defaultdict
import gymnasium as gym
import logging
import numpy as np
from pprint import pprint

logger = logging.getLogger("ray")

from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.models.configs import MLPEncoderConfig, MLPHeadConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.core.models.configs import ActorCriticEncoderConfig


class MyTorchPPORLModule(PPOTorchRLModule):
    """
    A PPORLModule made to learn RLModules API.
    """

    def setup(self):
        default_config = defaultdict(lambda: None)
        default_config["post_fcnet_hiddens"] = []

        for k,v in self.config.model_config_dict.items():
            default_config[k] = v

        logger.warning(str(self.config))
        logger.warning("START MyTorchPPORLModule")
        encoder_config = MLPEncoderConfig(
            input_dims=self.config.observation_space.shape,
            hidden_layer_dims=default_config["fcnet_hiddens"],
            hidden_layer_activation=default_config["fcnet_activation"],
        )
        logger.warning("Made Encoder Config " +\
            str(encoder_config.hidden_layer_activation))
        # Since we want to use PPO, which is an actor-critic algorithm, we need to
        # use an ActorCriticEncoderConfig to wrap the base encoder config.
        actor_critic_encoder_config = ActorCriticEncoderConfig(
            base_encoder_config=encoder_config
        )
        logger.warning("Made Actor-Critic Config")

        self.encoder = actor_critic_encoder_config.build(framework="torch")
        encoder_output_dims = encoder_config.output_dims
        logger.warning("Built Actor-Critic Encoder")

        pi_config = MLPHeadConfig(
            input_dims=encoder_output_dims,
            hidden_layer_dims=default_config["post_fcnet_hiddens"],
            output_layer_dim=int(self.config.action_space.n),
            hidden_layer_activation=default_config["post_fcnet_activation"],
        )
        logger.warning("Made Pi Config " +\
            str(pi_config.hidden_layer_activation))

        vf_config = MLPHeadConfig(
            input_dims=encoder_output_dims, 
            hidden_layer_dims=default_config["post_fcnet_hiddens"],
            output_layer_dim=1,
            hidden_layer_activation=default_config["post_fcnet_activation"],
        )
        logger.warning("Made Vf Config " +\
            str(vf_config.hidden_layer_activation))

        self.pi = pi_config.build(framework="torch")
        self.vf = vf_config.build(framework="torch")
        logger.warning("Built Heads")

        self.action_dist_cls = TorchCategorical
        logger.warning("END MyTorchPPORLModule")

config = (
    PPOConfig()
    .rl_module(_enable_rl_module_api=True,
        rl_module_spec=SingleAgentRLModuleSpec(module_class=MyTorchPPORLModule,
        model_config_dict={ "fcnet_hiddens" : [64],
            "fcnet_activation": "tanh",
            #"post_fcnet_hiddens": [16],
            "post_fcnet_activation": "relu",
        })
    )
    .environment("CartPole-v1")
    # The following training settings make it so that a training iteration is very
    # quick. This is just for the sake of this example. PPO will not learn properly
    # with these settings!
    .training(_enable_learner_api=True,
        train_batch_size=32, sgd_minibatch_size=16, num_sgd_iter=1)
)

