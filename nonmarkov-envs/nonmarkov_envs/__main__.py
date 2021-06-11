"""Main script: raises an error."""
from nonmarkov_envs.rdp_env import RDPEnv
from nonmarkov_envs.specs.rotating_maze import RotatingMaze
#from nonmarkov_envs.mcts import MonteCarloTreeSearchNode



def main():   
    env_spec = RotatingMaze()
    env = RDPEnv(env_spec, markovian=False, stop_prob=0.01)
    env.reset()

    print(env.theta(state= (0, 0, 0, 0)))
    print(env.tau(state= (0, 0, 0, 0)))
    # dict_action = env.theta(state= (0, 0, 0, 0))
    
    #print(env.theta(state=(2, 3)))
    '''print(env.step(0))
    print(env.step(1))
    print(env.step(2))
    print(env.step(3))'''
    #print(env.tau())
    #env.theta(state=(0, 0, 0, 0))
    #print(env._compute_action_space(new_theta))
    #root = MonteCarloTreeSearchNode(state = initial_state)
    #selected_node = root.best_action()
    return
    
if __name__ == "__main__":
    main()
