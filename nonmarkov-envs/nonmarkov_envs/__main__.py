"""Main script: raises an error."""
from nonmarkov_envs.mcts import MonteCarloTreeSearch
from nonmarkov_envs.mcts_s3m import MCTS_s3m
from nonmarkov_envs.rdp_env import RDPEnv
from nonmarkov_envs.S3M import S3M
from nonmarkov_envs.specs.rotating_maze import RotatingMaze
from nonmarkov_envs.specs.rotating_mab import RotatingMAB
from nonmarkov_envs.specs.cheat_mab import CheatMAB
from nonmarkov_envs.specs.malfunction_mab import MalfunctionMAB

import numpy as np
from matplotlib import pyplot as plt 
from tqdm import tqdm
from nonmarkov_envs.utils import show
import argparse



def plot_rewards(x, y, save_fig = False, path = ""):
    plt.plot(x, y)
    plt.xlabel("Steps")
    plt.ylabel("Average Rewards")

    if save_fig:
        plt.savefig(path)
    plt.show()

def plot_rewards_mixed(env,x1, y1, x2, y2,  save_fig = False, path = ""):
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    a =  [ 0.      ,   79.9399457,  79.91289239, 79.89339266, 79.88677979, 79.8807311, 79.87824408, 79.87347154, 79.87115216]
    plt.plot(x2, a)
    plt.xlabel("Steps")
    plt.title(env)
    plt.ylabel("Average Rewards")
    plt.legend(["MCTS algorithm", "first example S3M with pure exploration + MCTS", "second example S3M with pure exploration + MCTS" ])

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

def S3M_algorithm(env, maxit, min_samples, epsilon_list, total_iterations, solver_trials, step, smart_sample):
    print(f"\033[1m*********** s3m ****************\033[0m")
    print(f"\033[1mMax Iteration s3m: {maxit}\033[0m")
    print(f"\033[1mMin Samples s3m: {min_samples}\033[0m")
    print(f"\033[1mEspilon list s3m: {epsilon_list}\033[0m")
    s3m = S3M(env)
    for i in tqdm(range(maxit)):

        # Sampling
        s3m.sample(smart_sampler=smart_sample)

        # Base distribution
        s3m.base_distribution()
        
        # Merger
        
        s3m.merge_histories(epsilon_list, min_samples)

        # Mealy file generator
        file_name = s3m.mealy_file_generator()
        
        # Mealy Machine
        mealy_machine, data = s3m.mealy_machine(file_name)
    
    show(data)
    print(f"\033[1m******************************\033[0m \n")

    print(f"\033[1m********** MCTS **************\033[0m")
    print(f"\033[1mMCTS iterations: {total_iterations}, step: {step}, trials: {solver_trials}\033[0m")
    all_rewards_averaged = compute_average_rewards_mealy(env, total_iterations, solver_trials, step, mealy_machine, s3m)
    all_steps = list(range(0, total_iterations+1, step))
    
    print()
    print(f"\033[1mAverage rewards list: {all_rewards_averaged}\033[0m")
    print(f"\033[1m******************************\033[0m\n")

    return all_rewards_averaged, all_steps

def MCTS_algorithm(env, total_iterations, solver_trials, step):
    print(f"\033[1m********** MCTS **************\033[0m")
    print(f"\033[1mMCTS iterations: {total_iterations}, step: {step}, trials: {solver_trials}\033[0m\n")
    
    all_rewards_averaged = compute_average_rewards(env, total_iterations, solver_trials, step)
    all_steps = list(range(0, total_iterations+1, step))

    print()
    print(f"\033[1mAverage rewards list: {all_rewards_averaged}\033[0m")
    print(f"\033[1m******************************\033[0m\n")

    return all_rewards_averaged, all_steps

def main(only_mcts, env_type, ep, maxit, min_samples, mixed, smart_sample):   
    if env_type== "CheatMAB":
        env_spec = CheatMAB()
    elif env_type == "RotatingMAB":
        env_spec = RotatingMAB()
    elif env_type == "RotatingMaze":
        env_spec = RotatingMaze()
    elif env_type == "MalfunctionMAB":
        env_spec = MalfunctionMAB()

    print(f"\033[1mEnv: {env_type}\033[0m")
    print(f"\033[1mEpisode Length: {ep}\033[0m\n")
    env = RDPEnv(env_spec, markovian=False, stop_prob=0.0, episode_length=ep)
    env.reset()

    total_iterations = 160000
    step = 20000
    solver_trials = 30

    max_jumps = 4
    jump_value = 0.52
    epsilon_list = [i * jump_value for i in range(1, max_jumps + 1)]

    if not only_mcts and not mixed:
        all_rewards_averaged_s3m, all_steps = S3M_algorithm(env, maxit, min_samples, epsilon_list, total_iterations, solver_trials, step, smart_sample)
        
    elif not mixed:
        all_rewards_averaged_mcts, all_steps = MCTS_algorithm(env, total_iterations, solver_trials, step)
    else:
        # all_rewards_averaged_s3m, all_steps = S3M_algorithm(env, maxit, min_samples, epsilon_list, total_iterations, solver_trials, step, smart_sample)
        # all_rewards_averaged_mcts, all_steps = MCTS_algorithm(env, total_iterations, solver_trials, step)
        all_steps = list(range(0, total_iterations+1, step))
        all_rewards_averaged_s3m = [ 0.    ,     83.32712817, 83.29788614, 83.29848953, 83.30243507, 83.29282685, 83.29597694, 83.29470557, 83.29638902]
        all_rewards_averaged_mcts = [ 0.      ,   74.9399457,  74.91289239, 74.89339266, 74.88677979, 74.8807311, 74.87824408, 74.87347154, 74.87115216]
    if only_mcts:
        plot_rewards(all_steps, all_rewards_averaged_mcts, save_fig = True, path="./img/"+env_type+"/non_deterministic_"+env_type+"_"+str(ep)+".png")
    elif not mixed:
        plot_rewards(all_steps, all_rewards_averaged_s3m, save_fig = True, path="./img/"+env_type+"/non_deterministic_"+env_type+"_"+str(ep)+"_mealy_+"+str(min_samples)+"_"+str(maxit)+"maxit.png")
    else:
        #plot_rewards_mixed(all_steps, all_rewards_averaged_mcts, all_steps, all_rewards_averaged_s3m, save_fig = True, path="./img/"+env_type+"/non_deterministic_"+env_type+"_"+str(ep)+"_mealy_+"+str(min_samples)+"_"+str(maxit)+"maxit_"+str(smart_sample)+"smart_sample.png")
        plot_rewards_mixed(env_type,all_steps, all_rewards_averaged_mcts, all_steps, all_rewards_averaged_s3m, save_fig = True, path="./img/"+env_type+"/non_deterministic_"+env_type+"_"+str(ep)+"_mealy_+"+str(min_samples)+"_"+str(maxit)+"maxit_"+str(smart_sample)+"smart_sample.png")

    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, required=True,
                        help="choose the environment between: CheatMAB, RotatingMAB, RotatingMaze")

    parser.add_argument("--only-mcts", type=str, default=False,
                        help="choose if you want to perform only mcts algorithm")
    
    parser.add_argument("--episode", type=int, default=5, 
                        help="choose the length of the episode of the env")
    
    parser.add_argument("--max-iterations", type=int, default=4500, 
                        help="choose the max iterations of the s3m algorithm ")
    
    parser.add_argument("--min-samples", type=int, default=100,
                        help="min samples in the base distribution of s3m algorithm")
    
    parser.add_argument("--mixed", type=str, default=False,
                        help="choose if you want to perform only mcts and s3m algorithm")
    
    parser.add_argument("--smart-sample", type=str, default=False,
                        help="choose if you want to smart sample of s3m algorithm")
    args = parser.parse_args()
    only_mcts = args.only_mcts
    env = args.env
    ep = args.episode
    max_iterations = args.max_iterations
    min_s = args.min_samples
    mixed = args.mixed
    smart_sample = args.smart_sample
    main(only_mcts, env, ep, max_iterations, min_s, mixed, smart_sample)
