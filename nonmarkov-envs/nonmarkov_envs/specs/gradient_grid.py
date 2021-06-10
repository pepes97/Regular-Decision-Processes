from itertools import product
from math import ceil


class GradientGrid():
    """
    Specification of a grid environment with gradient rewards.
    """
    VERSION = 0.1

    def __init__(
        self,
        n: int = 5,
        m: int = 5
    ):
        # Actions
        self.UP = 0
        self.LEFT = 1
        self.DOWN = 2
        self.RIGHT = 3
        self.ACTIONS = set([self.UP, self.LEFT, self.DOWN, self.RIGHT])

        # States
        self.GRID_N = n
        self.GRID_M = m
        self.POSITIONS = set([(n, m) for n, m in product(range(self.GRID_N), range(self.GRID_M))])
        self.STATES = self.POSITIONS
        self.INITIAL_POSITION = tuple([0, 0])
        self.GOAL_POSITION = tuple([ceil((self.GRID_N-1)/2), ceil((self.GRID_M-1)/2)])
        self.MAX_DISTANCE = self._compute_distance(self.INITIAL_POSITION, self.GOAL_POSITION)

        # Observations
        self.OBSERVATIONS = set([(action, x, y) for action, (x, y) in product(self.ACTIONS, self.POSITIONS)])  

        # Rewards
        self.REWARDS = set([self._compute_reward(position) for position in self.POSITIONS])

        # Specify the initial state
        self.initial_state = tuple([0, 0])

        # Specify terminal states (if any)
        self.terminal_states = set([self.GOAL_POSITION])

        # Specify transition function and output function
        self.tau = self._compute_tau()
        self.theta = self._compute_theta()

    def _compute_reward(self, position):
        distance_to_goal = self._compute_distance(position, self.GOAL_POSITION)
        reward = self.MAX_DISTANCE - distance_to_goal
        return reward
    
    def _compute_distance(self, p1, p2):
        p1_x, p1_y = p1
        p2_x, p2_y = p2
        distance = abs(p1_x - p2_x) + abs(p1_y - p2_y)
        return distance

    def _next_position_is_valid(self, prev_pos, next_pos):
        prev_x = prev_pos[0]
        prev_y = prev_pos[1]
        move_up = (
            (next_pos == (prev_x, prev_y + 1)) or 
            (next_pos == prev_pos and prev_y == self.GRID_N-1)
        )
        move_left = (
            (next_pos == (prev_x - 1, prev_y)) or
            (next_pos == prev_pos and prev_x == 0)
        )
        move_down = (
            (next_pos == (prev_x, prev_y - 1)) or
            (next_pos == prev_pos and prev_y == 0)
        )
        move_right = (
            (next_pos == (prev_x + 1, prev_y)) or
            (next_pos == prev_pos and prev_x == self.GRID_M-1)
        )
        return move_up or move_left or move_down or move_right

    def _transition_is_valid(self, prev_pos, next_pos, action):
        prev_x = prev_pos[0]
        prev_y = prev_pos[1]

        move_up = (
            (action == self.UP and next_pos == (prev_x, prev_y + 1)) or 
            (action == self.UP and next_pos == prev_pos and prev_y == self.GRID_N-1)
        )
        move_left = (
            (action == self.LEFT and next_pos == (prev_x - 1, prev_y)) or
            (action == self.LEFT and next_pos == prev_pos and prev_x == 0)
        )
        move_down = (
            (action == self.DOWN and next_pos == (prev_x, prev_y - 1)) or
            (action == self.DOWN and next_pos == prev_pos and prev_y == 0)
        )
        move_right = (
            (action == self.RIGHT and next_pos == (prev_x + 1, prev_y)) or
            (action == self.RIGHT and next_pos == prev_pos and prev_x == self.GRID_M-1)
        )
        return move_up or move_left or move_down or move_right

    def _compute_tau(self):
        tau = dict()
        for state in self.STATES:
            tau[state] = dict()
            state_pos_x = state[0]
            state_pos_y = state[1]
            for observation in self.OBSERVATIONS:
                action = observation[0]
                next_pos_x = observation[1]
                next_pos_y = observation[2]
                if self._transition_is_valid((state_pos_x, state_pos_y), (next_pos_x, next_pos_y), action):
                    next_state = (next_pos_x, next_pos_y) 
                    tau[state][observation] = next_state
        return tau

    def _compute_theta(self):
        theta = dict()
        for state in self.STATES:
            theta[state] = dict()
            state_pos_x = state[0]
            state_pos_y = state[1]
            for action in self.ACTIONS:
                theta[state][action] = dict()
                for observation in self.OBSERVATIONS:
                    obs_action = observation[0]
                    obs_pos_x = observation[1]
                    obs_pos_y = observation[2]
                    if action == obs_action:
                        if self._transition_is_valid((state_pos_x, state_pos_y), (obs_pos_x, obs_pos_y), action):
                            theta[state][action][observation] = dict()
                            reward = self._compute_reward((obs_pos_x, obs_pos_y))
                            theta[state][action][observation][reward] = 1.0
        return theta


def instantiate_env(env_params):
    return GradientGrid(**env_params)
