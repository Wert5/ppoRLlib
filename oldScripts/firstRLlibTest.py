import torch
from pprint import pprint

from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .framework("torch")
    .environment("CartPole-v1")
    .rl_module(_enable_rl_module_api=True)
    .training(_enable_learner_api=True)
)

algorithm = config.build()

# run for 2 training steps
for _ in range(2):
    result = algorithm.train()
    pprint(result)


