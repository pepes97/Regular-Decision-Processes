

class CheatMAB():
    """
    Specification for the Cheat MAB domain.

    Reference:
    Abadi and Brafman (2020). "Learning and Solving Regular Decision Processes."
    """
    VERSION = 0.1

    def __init__(
        self,
        nb_arms: int = 2,
        win_probs: list = [0.2, 0.2],
        cheat_sequence: list = [0, 0, 0, 1],
        reward_win: int = 100,
        initial_state: tuple = (0,)
    ):
        # Cheat sequence
        self.cheat_sequence = cheat_sequence

        # States and actions
        self.NB_ARMS = nb_arms
        self.STATES = set([(seq_index,) for seq_index in range(len(cheat_sequence) + 1)])
        self.ACTIONS = set(range(nb_arms))

        # Observations
        self.LOSE = 0
        self.WIN = 1
        self.OBSERVATIONS = set([(action,) for action in self.ACTIONS])  

        # Observation probabilities
        self.WIN_PROBABILITIES = win_probs

        # Rewards
        self.REWARD_LOSE = 0
        self.REWARD_WIN = reward_win
        self.REWARDS = set([self.REWARD_LOSE, self.REWARD_WIN])

        # Specify the initial state
        self.initial_state = initial_state

        # Specify terminal states (if any, otherwise an empty set)
        self.terminal_states = set()

        # Specify transition function and output function
        self.tau = self._compute_tau() 
        self.theta = self._compute_theta()

    def _compute_tau(self):
        tau = dict()
        for state in self.STATES:
            tau[state] = dict()
            for observation in self.OBSERVATIONS:
                action = observation[0]
                if state[0] == len(self.cheat_sequence):
                    next_state = state[0]
                    tau[state][observation] = (next_state,)
                else:
                    for i, cheat_arm in enumerate(self.cheat_sequence):
                        if state[0] == i:
                            if action == cheat_arm:
                                next_state = state[0] + 1
                                break
                            else:
                                next_state = 0
                                break
                    tau[state][observation] = (next_state,)
        return tau

    def _compute_theta(self):
        theta = dict()
        for state in self.STATES:
            theta[state] = dict()
            for action in self.ACTIONS:
                theta[state][action] = dict()
                for observation in self.OBSERVATIONS:
                    if action == observation[0]:
                        if state[0] == len(self.cheat_sequence):
                            theta[state][action][observation] = dict()
                            theta[state][action][observation][self.REWARD_WIN] = 1.0
                            theta[state][action][observation][self.REWARD_LOSE] = 0.0
                        else:
                            theta[state][action][observation] = dict()
                            theta[state][action][observation][self.REWARD_WIN] = self.WIN_PROBABILITIES[action]
                            theta[state][action][observation][self.REWARD_LOSE] = 1 - self.WIN_PROBABILITIES[action]
        return theta


def instantiate_env(env_params):
    return CheatMAB(**env_params)
