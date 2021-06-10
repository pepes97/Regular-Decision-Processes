"""Main script: raises an error."""
from nonmarkov_envs.rdp_env import RDPEnv
from nonmarkov_envs.specs.rotating_maze import RotatingMaze
#from nonmarkov_envs.mcts import MonteCarloTreeSearchNode



def main():   
    env_spec = RotatingMaze()
    #print(env_spec.theta[(2, 3, 0, 0)])
    #print(env_spec.GOAL_POSITION)
    # pippo
    # env = RDPEnv(env_spec, markovian=False, stop_prob=0.01)
    # print(env.theta(state= (0, 0, 0, 0)))
    # dict_action = env.theta(state= (0, 0, 0, 0))
    
    #print(env.theta(state=(2, 3)))
    print(RDPEnv(env_spec, markovian=False, stop_prob=0.01).step(0))
    print(RDPEnv(env_spec, markovian=False, stop_prob=0.01).step(1))
    print(RDPEnv(env_spec, markovian=False, stop_prob=0.01).step(2))
    print(RDPEnv(env_spec, markovian=False, stop_prob=0.01).step(3))
    #print(env.tau())
    #env.theta(state=(0, 0, 0, 0))
    #print(env._compute_action_space(new_theta))
    #root = MonteCarloTreeSearchNode(state = initial_state)
    #selected_node = root.best_action()
    return
    
if __name__ == "__main__":
    main()
