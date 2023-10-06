from pprint import pprint
from ray.rllib.policy import Policy
import gymnasium as gym
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

def main():
    num_eval_episodes = 100

    iters = []
    means = []

    for i in range(2,41,2):
        policy = Policy.from_checkpoint(
            "/home/winstongrenier/rlPPO1/breakoutChecks1/iter" + str(i)\
            + "/policies/default_policy")

        # instantiate env class
        env = gym.vector.make("CartPole-v1", num_envs=num_eval_episodes)

        # run until episode ends
        episode_reward = 0
        done = False
        obs, infos = env.reset()
        dones = np.zeros(num_eval_episodes, dtype=bool)
        while not all(dones):
            actions, states, infos = policy.compute_actions(obs)
            obs, rewards, terminates, truncates, infos = env.step(actions)
            # Add up rewards for episodes that have not completed
            episode_reward += sum([rewards[i] for i in range(rewards.shape[0])
                if not dones[i]])
            # Update flags indicating which episodes are done
            dones = np.logical_or(terminates, dones)

        print("\nEVALUATION ITER", i)
        mean_reward = episode_reward / num_eval_episodes
        print("Mean Reward:", mean_reward)
        iters.append(i)
        means.append(mean_reward)


    print(iters)
    print(means)

    plt.plot(iters, means)
    plt.title("Reward vs Iterations")
    plt.ylabel("Mean Reward")
    plt.xlabel("Iteration")
    plt.savefig("/home/winstongrenier/rlPPO1/iterMeanPlot.png")

if __name__ == "__main__":
    main()

