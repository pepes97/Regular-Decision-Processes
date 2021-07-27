from collections import deque


class RotatingMAB():
    """
    Specification for the Rotating MAB domain.

    Reference:
    Abadi and Brafman (2020). "Learning and Solving Regular Decision Processes."
    """
    VERSION = 0.1

    def __init__(
        self,
        nb_arms: int = 2,
        win_probs: list = [0.9, 0.2],
        reward_win: int = 100,
        initial_state: tuple = (0,)
    ):
        # States and actions
        self.NB_ARMS = nb_arms
        self.STATES = set([(arm,) for arm in range(nb_arms)])
        self.ACTIONS = set(range(nb_arms))

        # Observations
        self.LOSE = 0
        self.WIN = 1
        self.OBSERVATIONS = set([tuple([self.LOSE]), tuple([self.WIN])])

        # Observation probabilities
        self.WIN_PROBABILITIES = win_probs

        # Rewards
        self.REWARD_LOSE = 0
        self.REWARD_WIN = reward_win
        self.REWARDS = set([self.REWARD_LOSE, self.REWARD_WIN])

        # Specify the initial state
        self.initial_state = initial_state

        # Specify terminal states (if any)
        self.terminal_states = set()

        # Specify transition function and output function
        self.tau = self._compute_tau()
        self.theta = self._compute_theta()

    def _compute_tau(self):
        tau = dict()
        for state in self.STATES:
            tau[state] = dict()
            for observation in self.OBSERVATIONS:
                if observation == tuple([self.WIN]):
                    next_state = (state[0] + 1) % self.NB_ARMS
                    tau[state][observation] = (next_state,)
                else:
                    tau[state][observation] = state
        return tau

    def _compute_theta(self):
        theta = dict()
        for state in self.STATES:
            theta[state] = dict()
            win_probs = deque(self.WIN_PROBABILITIES)
            win_probs.rotate(state[0])
            for action in self.ACTIONS:
                theta[state][action] = dict()
                win_prob = win_probs[action]
                for observation in self.OBSERVATIONS:
                    theta[state][action][observation] = dict()
                    if observation == tuple([self.WIN]):
                        theta[state][action][observation][self.REWARD_WIN] = win_prob
                    else:
                        theta[state][action][observation][self.REWARD_LOSE] = 1 - win_prob
        return theta


def instantiate_env(env_params):
    return RotatingMAB(**env_params)
