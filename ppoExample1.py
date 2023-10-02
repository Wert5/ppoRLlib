from pprint import pprint
from ray.rllib.algorithms.ppo import PPOConfig

config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment("BreakoutNoFrameskip-v4")
    .resources(num_gpus=1, num_gpus_per_learner_worker=1)
    .rollouts(num_rollout_workers=4)
    .framework("torch")
    .training(model={"fcnet_hiddens": [64, 64, 64, 64]},
        lambda_=0.95, kl_coeff=0.5, clip_param=0.1, vf_clip_param=10.0,
        entropy_coeff=0.01, train_batch_size=1000, num_sgd_iter=10,
        lr_schedule=[[0,1e-3],[1000, 0.0]])
    .evaluation(evaluation_num_workers=1, evaluation_duration=5)
)

algo = config.build()  # 2. build the algorithm,

for i in range(10):
    print("Start iter", i)
    (algo.train())  # 3. train it,

print("\nEVALUATION")
pprint(algo.evaluate())  # 4. and evaluate it.
