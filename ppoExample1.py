from pprint import pprint
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms import Algorithm

max_timesteps = int(1e1)

config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment("BreakoutNoFrameskip-v4",
        clip_rewards=True, clip_actions=True)
    .resources(num_gpus=1, num_gpus_per_learner_worker=1)
    .rollouts(num_rollout_workers=10, num_envs_per_worker=5,
        observation_filter="NoFilter",
        batch_mode="truncate_episodes")
    .framework("torch")
    .training(model={"vf_share_layers": True, "conv_filters": None,
            "conv_activation": "relu", "fcnet_activation": "tanh",
            "fcnet_hiddens": [256, 256], "free_log_std": False,
            "no_final_linear": False, "use_lstm": False, "framestack": True,
            "dim": 84, "grayscale": False, "zero_mean": True,
            "custom_model": None},
        entropy_coeff=0.01, train_batch_size=5000, num_sgd_iter=10,
        lr_schedule=[[0, 5e-5]],
        vf_clip_param=10.0, kl_coeff=0.5, lambda_=0.95,
        clip_param=0.1, sgd_minibatch_size=500)
    .evaluation(evaluation_num_workers=2, evaluation_duration=10,
        evaluation_interval=10)
)

algo = config.build()  # 2. build the algorithm,

#algo = Algorithm.from_checkpoint("/home/winstongrenier/rlPPO1/breakoutChecks1/iter")

for i in range(100):
    print("Start iter", i)
    train_dict = algo.train()  # 3. train it,
    print("Time Total s:", train_dict["time_total_s"])
    print("Time This Iter s:", train_dict["time_this_iter_s"])
    iter_num = train_dict["training_iteration"]
    print("Iteration:", iter_num)
    if "evaluation" in train_dict:
        pprint(train_dict["evaluation"]["sampler_results"])
        save_res = algo.save(checkpoint_dir=\
            "/home/winstongrenier/rlPPO1/breakoutChecks1/iter" + str(iter_num))
        print("SAVED AT", save_res.checkpoint.path)

print("\nEVALUATION FINAL")
pprint(algo.evaluate())  # 4. and evaluate it.
save_res = algo.save(checkpoint_dir=\
    "/home/winstongrenier/rlPPO1/breakoutChecks1/finalCheck")
print("SAVED AT", save_res.checkpoint.path)

print("\nWEIGHTS")
pprint({ p : {k : v.shape for k, v in w.items()}
    for p, w in algo.get_weights().items()})
