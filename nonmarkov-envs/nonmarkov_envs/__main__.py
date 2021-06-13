"""Main script: raises an error."""
from nonmarkov_envs.mcts import MonteCarloTreeSearch
from nonmarkov_envs.rdp_env import RDPEnv
from nonmarkov_envs.specs.rotating_maze import RotatingMaze
#from nonmarkov_envs.mcts import MonteCarloTreeSearchNode
import sys

sys.setrecursionlimit(1000000)

def main():   
    env_spec = RotatingMaze()
    env = RDPEnv(env_spec, markovian=False, stop_prob=0.0, episode_length=100)
    env.reset()

    '''print(env.theta(state= (0, 0, 0, 0)))
    print(env.tau(state= (0, 0, 0, 0)))
    # dict_action = env.theta(state= (0, 0, 0, 0))'''
    
    mcts = MonteCarloTreeSearch(env)
    mcts_initial_state = mcts.mcts(1000)

    mcts.print_best_path(mcts_initial_state, False)
    return
    
if __name__ == "__main__":
    main()
