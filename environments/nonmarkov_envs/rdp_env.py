import numpy as np
from gym.spaces import Discrete, MultiDiscrete
from collections import defaultdict
from functools import partial
from .discrete_env import DiscreteEnv, space_size


class RDPEnv(DiscreteEnv):
    """
    RDP environment generated out of specifications of transition function tau and output function theta.
    """
    VERSION = 0.2

    def __init__(
        self, 
        specification, 
        markovian: bool = False, 
        stop_prob: float = 0.05, 
        episode_length: int = None, 
        rand_seed: int = 0 
    ): 
        self.specification = specification
        self.markovian = markovian

        # Set stop probability or fixed episode length (if not None)
        if episode_length is None:
            self.stop_prob = stop_prob
        else:
            self.episode_length = episode_length

        # Compute space of observations and states
        (
            observation_state_space, 
            observation_space, 
            state_space, 
            empty_observation, 
            empty_state, 
            observations, 
            states, 
            obs_length, 
            state_length 
        ) = self._compute_observation_space(self.tau)
        if markovian: 
            self.observation_space = state_space
            self.observations = states
        else:
            self.observation_space = observation_space
            self.observations = observations
        self._empty_observation = empty_observation
        self._empty_state = empty_state
        self._state_length = state_length
        self._obs_length = obs_length
        
        # Encoder/decoder for state symbols
        self.encoder = partial(np.ravel_multi_index, dims=observation_state_space.nvec)
        self.decoder = partial(np.unravel_index, shape=observation_state_space.nvec)

        # Compute space of actions
        action_space, actions = self._compute_action_space(self.theta)
        self.action_space = action_space
        self.actions = actions

        # Compute dynamics
        P, P_initial_state = self._compute_dynamics(self.tau, self.theta, specification.initial_state, specification.terminal_states)

        # Init super and set seed
        nS = space_size(observation_state_space)
        nA = space_size(action_space)
        ids = np.zeros(nS)
        ids[P_initial_state] = 1.0
        super().__init__(nS, nA, P, ids)
        self.seed(rand_seed)

    def _compute_observation_space(self, tau):
        """Compute the space of observations and actual states."""
        states = set()
        observations = set()
        state_elements = defaultdict(set)
        obs_elements = defaultdict(set)
        for state in tau():
            states.add(state)
            for index, state_element in enumerate(state):
                    state_elements[index].add(state_element)
            for observation in tau(state):
                observations.add(observation)
                for index, obs_element in enumerate(observation):
                    obs_elements[index].add(obs_element)
        # Get the number of elements in each state and observation tuples
        nb_state_element_values = tuple([len(values) for values in state_elements.values()])
        nb_obs_element_values = tuple([len(values) for values in obs_elements.values()])
        state_length = len(nb_state_element_values)
        obs_length = len(nb_obs_element_values)
        empty_state  = tuple(np.full(state_length, -1).tolist())
        empty_observation  = tuple(np.full(obs_length, -1).tolist())
        # Generate the spaces
        observation_state_space = MultiDiscrete(nb_obs_element_values + nb_state_element_values)
        observation_space = MultiDiscrete(nb_obs_element_values)
        state_space = MultiDiscrete(nb_obs_element_values)
        return (
            observation_state_space, 
            observation_space, 
            state_space, 
            empty_observation, 
            empty_state, 
            observations, 
            states, 
            obs_length, 
            state_length
        )

    def _compute_action_space(self, theta):
        """Compute the action space."""
        # Extract all possible actions from each state
        actions = set()
        for state in theta():
            for action in theta(state):
                actions.add(action)
        nb_actions = len(actions)
        return Discrete(nb_actions), actions

    def _compute_dynamics(self, tau, theta, initial_state, terminal_states):
        """Compute the dynamics of system."""
        P = dict()
        P_initial_state = None
        for state in tau():
            for current_obs in tau(state):
                current_state = tau(state, current_obs)
                P_state = current_obs + current_state
                P_state = self.encoder(P_state)
                # Store the initial state
                if current_state == initial_state:
                    encoded_initial_state = P_state
                P[P_state] = dict()
                for action in theta(current_state):
                    P[P_state][action] = set()
                    for next_obs in theta(current_state, action):
                        next_state = tau(current_state, next_obs)
                        P_next_state = next_obs + next_state
                        P_next_state = self.encoder(P_next_state)
                        for reward in theta(current_state, action, next_obs):
                            prob = theta(current_state, action, next_obs, reward)
                            if prob > 0:
                                # If episode length is not set, then use stop probabilities
                                if self.episode_length is None:
                                    # Done if terminal
                                    if next_state in terminal_states: 
                                        P[P_state][action].add((prob, P_next_state, reward, True))
                                    else:
                                        P[P_state][action].add((prob * self.stop_prob, P_next_state, reward, True))
                                        P[P_state][action].add((prob * (1-self.stop_prob), P_next_state, reward, False))
                                else:
                                    # Done if terminal
                                    if next_state in terminal_states: 
                                        P[P_state][action].add((prob, P_next_state, reward, True))
                                    else:
                                        P[P_state][action].add((prob, P_next_state, reward, False))
                        # Get actual initial state for P dynamics
                        if P_initial_state is None and next_state == initial_state:
                            P_initial_state = P_next_state
                    P[P_state][action] = list(P[P_state][action])
        return P, P_initial_state

    def tau(self, state=None, observation=None):
        """Get the values from the transition function specification."""
        if state is None:
            return self.specification.tau
        if observation is None:
            return self.specification.tau[state]
        else:
            return self.specification.tau[state][observation]

    def theta(self, state=None, action=None, observation=None, reward=None):
        """Get the values from the output function specification."""
        if state is None:
            return self.specification.theta
        elif action is None:
            return self.specification.theta[state]
        elif observation is None:
            return self.specification.theta[state][action]
        elif reward is None:
            return self.specification.theta[state][action][observation]
        else:
            return self.specification.theta[state][action][observation][reward]

    def _process(self, state):
        """Return actual state if Markovian, otherwise observation."""
        observation_and_state = list(self.decoder(state))
        if self.markovian: 
            return tuple(observation_and_state[self._obs_length:])
        else: 
            return tuple(observation_and_state[:self._obs_length])

    def _is_terminal(self, state):
        state = list(self.decoder(state))
        state = tuple(state[self._obs_length:])
        if state in self.specification.terminal_states:
            return True
        else:
            return False

    def reset(self, **kwargs):
        """Reset the environment."""
        state = super().reset(**kwargs) 
        if self.markovian: 
            state = self._process(state) 
        else: 
            state = self._empty_observation
        self.steps = 0
        return state

    def step(self, action):
        """Do a step."""
        state, reward, done, info = super().step(action)
        # Check if is a terminal state
        info['terminal_state'] = self._is_terminal(state)
        # Update number of steps and finish if episode length is set
        self.steps += 1
        if self.steps == self.episode_length:
            done = True
        # Process and return state
        new_state = self._process(state)
        return new_state, reward, done, info
