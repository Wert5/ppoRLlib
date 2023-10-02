import ray
from ray import train, tune
from ray.rllib.algorithms.ppo import PPO

def train_fn(config):
    algo = PPO(config=config).environment('CartPole-v1')
    while True:
        result = algo.train()
        train.report(result)
        if result["episode_reward_mean"] > 200:
            task = 2
        elif result["episode_reward_mean"] > 100:
            task = 1
        else:
            task = 0
        algo.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_task(task)))

num_gpus = 1
num_workers = 2

ray.init()
tune.Tuner(
    tune.with_resources(train_fn, resources=tune.PlacementGroupFactory(
        [{"CPU": 1}, {"GPU": num_gpus}] + [{"CPU": 1}] * num_workers
    ),),
    param_space={
        "num_gpus": num_gpus,
        "num_workers": num_workers,
    },
).fit()
