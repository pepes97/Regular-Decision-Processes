"""Main script: raises an error."""
from nonmarkov_envs.mcts import MonteCarloTreeSearch
from nonmarkov_envs.rdp_env import RDPEnv
from nonmarkov_envs.S3M import S3M
from nonmarkov_envs.specs.rotating_maze import RotatingMaze
import numpy as np
from matplotlib import pyplot as plt 
#from nonmarkov_envs.mcts import MonteCarloTreeSearchNode
import sys

sys.setrecursionlimit(100000)


def main():   
    env_spec = RotatingMaze()
    env = RDPEnv(env_spec, markovian=False, stop_prob=0.0, episode_length=15)
    env.reset()

    '''print(env.theta(state= (0, 0, 0, 0)))
    print(env.tau(state= (0, 0, 0, 0)))
    # dict_action = env.theta(state= (0, 0, 0, 0))'''
    
    total_iterations = 140000
    num_trials = 2

    all_rewards = []
    
    for i in range(num_trials):
        mcts = MonteCarloTreeSearch(env, total_iterations, False)
        mcts_initial_state, rewards = mcts.mcts(mcts.iterations)
        all_rewards.append(rewards)

    all_rewards_averaged = np.average(all_rewards, axis=0)
    print(all_rewards_averaged)

    plt.plot(list(range(0, total_iterations+1, 10000)), all_rewards_averaged)
    plt.xlabel("Steps")
    plt.ylabel("Average Rewards")
    plt.show()

    #mcts.print_best_path(mcts_initial_state, False)

    '''s3m = S3M(env)
    
    for i in range(200):
        s3m.sample()
        s3m.base_distribution(5)
        s3m.merger(1)
    print(s3m.max_dkl)'''
    #print(s3m.tr)
    #print(s3m.traces)
    # for k in s3m.traces:
    #     print(f"{k}:\n {s3m.traces[k]}\n\n")

    return
    
if __name__ == "__main__":
    main()
