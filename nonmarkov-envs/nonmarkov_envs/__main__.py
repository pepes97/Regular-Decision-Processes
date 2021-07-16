"""Main script: raises an error."""
from nonmarkov_envs.mcts import MonteCarloTreeSearch
from nonmarkov_envs.mcts_s3m import MCTS_s3m
from nonmarkov_envs.rdp_env import RDPEnv
from nonmarkov_envs.S3M import S3M
from nonmarkov_envs.specs.rotating_maze import RotatingMaze
import numpy as np
from matplotlib import pyplot as plt 
from tqdm import tqdm
from nonmarkov_envs.utils import show


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

def compute_average_rewards_mealy(env, total_iterations, num_trials, step, mealy_machine, s3m):
    all_rewards = []
    
    for i in range(num_trials):
        print(">> TRIAL ", i)
        mcts = MCTS_s3m(env, total_iterations, False, step, mealy_machine, s3m)
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
    # plot_rewards(all_steps, all_rewards_averaged, True, "./img/prova_today.png") # save plot
    # plot_rewards(all_steps, all_rewards_averaged)

    # mcts.print_best_path(mcts_initial_state, False)

    s3m = S3M(env)
    for i in tqdm(range(500)):

        # Sampling
        s3m.sample()

        # Base distribution
        s3m.base_distribution()
        
        # Merger
        s3m.merge_histories([0.2, 0.5, 3], 30)

        # Mealy file generator
        file_name = s3m.mealy_file_generator()
        
        # Mealy Machine
        mealy_machine, data = s3m.mealy_machine(file_name)
        
    total_iterations = 1000
    step = 200
    num_trials = 50

    all_rewards_averaged = compute_average_rewards_mealy(env, total_iterations, num_trials, step, mealy_machine, s3m)
    all_steps = list(range(0, total_iterations+1, step))
    
    plot_rewards(all_steps, all_rewards_averaged)
    return
    
if __name__ == "__main__":
    main()
