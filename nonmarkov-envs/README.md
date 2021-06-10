# Non-Markov Envs package

A collection of Non-Markovian environments and RDPs.

## Install
This package can be installed as usual:

    pip install .

Or, we can install a specific tested version of this package and its dependencies with:

    poetry install --no-dev

Omit the `--no-dev` option if you're installing for local development.

## Use

This package is not meant to be executed as a script, but only imported.

Usage example:

	from nonmarkov_envs.rdp_env import RDPEnv
	from nonmarkov_envs.specs.rotating_maze import RotatingMaze
	
	env_spec = RotatingMaze()
	env = RDPEnv(env_spec, markovian=False, stop_prob=0.01)

`env` is now a gym environment with non-Markovian observations.

Note: the interface might be changed/extended in the future. This is under
development. Prefer indicating specific package versions.

## Environments

Environments (domains) in this package formally specify RDPs (*A*, *O*, *R*, **D**), where:

- *A* is the finite set of actions;
- *O* is the finite set of observations;
- *R* is the finite set of non-negative rewards;

**D** is the dynamics function and is represented by a transducer (*S*, *s<sub>0</sub>*, *O*, &tau;, &theta;), as follows:

- *S* is the finite set of states;
- *s<sub>0</sub>* is the initial state *s<sub>0</sub>* &isin; *S*;
- *O* is the finite set of input symbols, i.e. RDP observations defined above;
- &tau; is the transition function that, given the current state and an observation, returns the next state;
- &theta; is the output function that, given the current state, specifies the probability of an action, observation, and reward.

In this package, environment specifications are classes that have the following attributes: 

	""" An environment specification.

	   ACTIONS: finite set of positive integers {0..n}.
	   STATES: finite set of tuples with non-negative element values {(0..n, 0..n, ..., 0..n), ...}.
	   OBSERVATIONS: set of tuples with non-negative element values {(0..n, 0..n, ..., 0..n), ...}.
	   REWARDS: set of non-negative rewards {0..n}.
	   initial_state: a state in STATES.
	   tau: the transition function, specified as a mapping.
	   theta: the output function, specified as a mapping.
	"""

Specifically, the transition function &tau; and the output function &theta; are represented by mappings with the following schema:

	tau = {
	   state: {
	      observation: next_state
	   }
	   ...
	}

	theta = {
	   state: {
	      action: 
	         observation:
	            reward: probability
	   }
	   ...
	}

Except for very simple domains, it is hard to specify tau and theta manually. In general, we write methods to generate these mappings (see examples in [``nonmarkov_envs/specs``](https://github.com/whitemech/nonmarkov-envs/tree/master/nonmarkov_envs/specs)).

## Contribute

You can make suggestions and propose fixes by opening a new issue in this repository.
If you would like to contribute, you may fork and open pull requests.
For new environment specifications, please consider following the specifications above.
