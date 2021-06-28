"""Main script: raises an error."""
from nonmarkov_envs.mcts import MonteCarloTreeSearch
from nonmarkov_envs.rdp_env import RDPEnv
from nonmarkov_envs.S3M import S3M
from nonmarkov_envs.specs.rotating_maze import RotatingMaze
import numpy as np
from matplotlib import pyplot as plt 
import sys

# sys.setrecursionlimit(100000)

def plot_rewards(x, y, save_fig = False, path = ""):
    plt.plot(x, y)
    plt.xlabel("Steps")
    plt.ylabel("Average Rewards")

    if save_fig:
        plt.savefig(path)
    plt.show()

def compute_average_rewards(env, total_iterations, num_trials, step):
    all_rewards = []
    
    for i in range(num_trials):
        print(">> TRIAL ", i)
        mcts = MonteCarloTreeSearch(env, total_iterations, False, step)
        mcts_initial_state, rewards = mcts.mcts(mcts.iterations)
        all_rewards.append(rewards)
    
    all_rewards_averaged = np.average(all_rewards, axis=0)
    return all_rewards_averaged

def main():   
    env_spec = RotatingMaze()
    env = RDPEnv(env_spec, markovian=False, stop_prob=0.0, episode_length=15)
    env.reset()
    
    # total_iterations = 140000
    # step = 20000
    # num_trials = 50

    # all_rewards_averaged = compute_average_rewards(env, total_iterations, num_trials, step)
    # all_steps = list(range(0, total_iterations+1, step))
    
    # print(all_rewards_averaged)
    # #plot_rewards(all_steps, all_rewards_averaged, True, "./img/prova50.png") # save plot
    # plot_rewards(all_steps, all_rewards_averaged)
    

    #mcts.print_best_path(mcts_initial_state, False)

    s3m = S3M(env)
    
    

    # print(s3m.traces)
    for i in range(5):
        s3m.sample()
        print(s3m.traces)
        s3m.base_distribution(1)
        print("Tr\n"+ str(s3m.tr) + "\n" + str(len(s3m.tr)))
        s3m.merge_histories([0.01])

    # for i in range(200):
    #     s3m.sample()
    #     s3m.base_distribution(5)
    #     s3m.merger(1)
    # print(s3m.max_dkl)
    
    #print(s3m.tr)
    #print(s3m.traces)
    # for k in s3m.traces:
    #     print(f"{k}:\n {s3m.traces[k]}\n\n")
    

    return
    
if __name__ == "__main__":
    main()
