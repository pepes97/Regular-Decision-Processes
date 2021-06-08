from itertools import product


class RewardPath():
    """
    Specification of a grid environment with a path of rewards.
    """
    VERSION = 0.1

    def __init__(
        self,
        n: int = 4,
        m: int = 4,
        reward: int = 100, 
        initial_state: tuple = (0, 0, 0, 1)
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
        self.STATES = set([(x1, y1, x2, y2)
                    for (x1, y1), (x2, y2) in product(self.POSITIONS, self.POSITIONS) if (x1,y1) != (x2,y2)])

        # Observations
        self.OBSERVATIONS = set([(x, y)
                    for x, y in product(range(self.GRID_N), range(self.GRID_M))])

        # Reward path (snake-like)
        self.path = list()
        for n in range(self.GRID_N):
            if n%2 == 0: self.path += [(n, y) for y in range(self.GRID_M)]
            else: self.path += list(reversed([(n, y) for y in range(self.GRID_M)]))

        # Rewards
        self.CELL_NO_REWARD = 0
        self.CELL_REWARDING = reward
        self.REWARDS = set([self.CELL_NO_REWARD, self.CELL_REWARDING])

        # Specify the initial state
        self.initial_state = initial_state

        # Specify terminal states (if any)
        terminal_state = (self.GRID_N-1, self.GRID_M-1, 0, 0)
        self.terminal_states = set([terminal_state])
        self.STATES.add((self.GRID_N-1, self.GRID_M-1, 0, 0))

        # Specify transition function and output function
        self.tau = self._compute_tau()
        self.theta = self._compute_theta()

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
            pos_x = state[0]
            pos_y = state[1]
            reward_x = state[2]
            reward_y = state[3]
            for observation in self.OBSERVATIONS:
                next_pos_x = observation[0]
                next_pos_y = observation[1]
                if self._next_position_is_valid((pos_x, pos_y), (next_pos_x, next_pos_y)):
                    if next_pos_x == reward_x and next_pos_y == reward_y:
                        # Get the next reward position
                        current_reward_index = self.path.index((reward_x, reward_y))
                        # If the reward is in the last grid cell of the path, then next reward goes to (0,0)
                        if len(self.path) == current_reward_index+1:
                            next_reward_x = 0
                            next_reward_y = 0
                        else:
                            next_reward_x, next_reward_y = self.path[current_reward_index+1]
                        next_state = (next_pos_x, next_pos_y, next_reward_x, next_reward_y) 
                    else:
                        next_state = (next_pos_x, next_pos_y, reward_x, reward_y) 
                    tau[state][observation] = next_state
        return tau

    def _compute_theta(self):
        theta = dict()
        for state in self.STATES:
            theta[state] = dict()
            pos_x = state[0]
            pos_y = state[1]
            reward_x = state[2]
            reward_y = state[3]
            for action in self.ACTIONS:
                theta[state][action] = dict()
                for observation in self.OBSERVATIONS:
                    obs_pos_x = observation[0]
                    obs_pos_y = observation[1]
                    if self._transition_is_valid((pos_x, pos_y), (obs_pos_x, obs_pos_y), action):
                        theta[state][action][observation] = dict()
                        if (obs_pos_x, obs_pos_y) == (reward_x, reward_y):
                            reward_prob = 1.0
                        else: 
                            reward_prob = 0.0
                        theta[state][action][observation][self.CELL_REWARDING] = reward_prob
                        theta[state][action][observation][self.CELL_NO_REWARD] = 1 - reward_prob
        return theta


def instantiate_env(env_params):
    return RewardPath(**env_params)
