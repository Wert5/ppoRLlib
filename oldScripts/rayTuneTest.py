from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune


algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=1)
    .environment(env="CartPole-v1")
    .framework("torch")
)

print("DONE BUILDING")

tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5},
        checkpoint_freq=10,
        storage_path="~/rlPPO/rayTuneCartPole1",
        config=algo.to_dict(),
    )

