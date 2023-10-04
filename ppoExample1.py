from pprint import pprint
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms import Algorithm

config = (  # 1. Configure the algorithm,
    PPOConfig()
    .rl_module(_enable_rl_module_api=True)
    .environment("BreakoutNoFrameskip-v4", clip_rewards=True,
        env_config={"frameskip" : 1, "full_action_space" : False,
        "repeat_action_probability" : 0.0})
    .resources(num_gpus=1, num_gpus_per_learner_worker=1)
    .rollouts(num_rollout_workers=10, num_envs_per_worker=5,
        rollout_fragment_length="auto")
    .framework("torch")
    .training(_enable_learner_api=True, model={"vf_share_layers": True,
            "conv_filters" : [[16, 4, 2], [32, 4, 2], [64, 4, 2], [128, 4, 2]],
            "conv_activation": "relu", "post_fcnet_hiddens": [256]},
        entropy_coeff=0.01, train_batch_size=4000, num_sgd_iter=10,
        lr_schedule=[[0, 1e-4]], vf_clip_param=10.0, kl_coeff=0.5,
        lambda_=0.95, clip_param=0.1, sgd_minibatch_size=1000,
        grad_clip=100.0, grad_clip_by="global_norm")
    .evaluation(evaluation_num_workers=2, evaluation_duration=10,
        evaluation_interval=10)
)

algo = config.build()  # 2. build the algorithm,

#algo = Algorithm.from_checkpoint("/home/winstongrenier/rlPPO1/breakoutChecks1/iter")

for i in range(10):
    print("Start iter", i)
    train_dict = algo.train()  # 3. train it,
    print("Time Total s:", train_dict["time_total_s"])
    print("Time This Iter s:", train_dict["time_this_iter_s"])
    iter_num = train_dict["training_iteration"]
    print("Iteration:", iter_num)
    if "evaluation" in train_dict:
        pprint(train_dict)
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
