import numpy as np
from collections import defaultdict

class MonteCarloTreeSearchNode():
    def __init__(self, state, parent=None, parent_action=None, _untried_actions=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = _untried_actions
        return


    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses
    
    def n(self):
        return self._number_of_visits

    def print_node(self):
        print(f"state: {self.state}\n parent_state: {self.parent}\n parent_action: {self.parent_action}")


    def __eq__(self, other):
        return self.state == other.state and self.parent_action == other.parent_action
    
   