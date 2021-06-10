from .mcts_node import MonteCarloTreeSearchNode

class MonteCarloTreeSearch():
    def __init__(self, env):
        self.env = env
        self.initial_state = MonteCarloTreeSearchNode(state = env.specification.initial_state)

    def step(self, state):
        theta = self.env_spec.theta[state]
        
        new_state, reward, done, info = self.env.step(action)
        return MonteCarloTreeSearchNode(state = new_state)