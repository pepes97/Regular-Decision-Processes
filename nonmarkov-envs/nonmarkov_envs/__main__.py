"""Main script: raises an error."""
from nonmarkov_envs.mcts import MonteCarloTreeSearch
from nonmarkov_envs.rdp_env import RDPEnv
from nonmarkov_envs.S3M import S3M
from nonmarkov_envs.specs.rotating_maze import RotatingMaze
#from nonmarkov_envs.mcts import MonteCarloTreeSearchNode
import sys

sys.setrecursionlimit(1000000)

def main():   
    env_spec = RotatingMaze()
    env = RDPEnv(env_spec, markovian=False, stop_prob=0.0, episode_length=15)
    env.reset()

    '''print(env.theta(state= (0, 0, 0, 0)))
    print(env.tau(state= (0, 0, 0, 0)))
    # dict_action = env.theta(state= (0, 0, 0, 0))'''
    
    mcts = MonteCarloTreeSearch(env, 1000)
    mcts_initial_state = mcts.mcts(mcts.iterations)

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
