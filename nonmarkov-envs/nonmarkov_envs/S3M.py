import random
import numpy as np

from numpy.random import normal

class S3M():
    def __init__(self, env):
        self.env = env
        self.initial_state = self.env.specification.initial_state
        self.traces = []
        self.transitions = []
    
    def sample(self, mode=0):
        ''' Sample a new trace by exploring the environment
        mode = 0 for pure Exploration
               1 for Smart Sampling (not yet implemented)
        '''
        current_trace = [self.initial_state]
        state = self.initial_state
        n_a_s = {}
        f_a_s = {}
        p_a_s = {}

        if mode == 0:
            def pureExploration(state, n_a_s, f_a_s, p_a_s):            
                theta = self.env.theta(state)
                
                if n_a_s.get(state) == None:
                    n_a_s[state] = [{}, 0]
                    f_a_s[state] = {}
                    p_a_s[state] = []
                    for action in list(theta.keys()):
                        n_a_s[state][0][action] = 0
                        f_a_s[state][action] = 0
                        p_a_s[state] = np.random.uniform(0,1,len(list(theta.keys())))
                
                selected_action = random.choices(list(theta.keys()), p_a_s[state])[0]
                n_a_s[state][0][selected_action] += 1
                n_a_s[state][1] += 1
                f_a_s[state][selected_action] = 1 - n_a_s[state][0][selected_action] / n_a_s[state][1]

                sumf_a_s = sum([f_a_s[state][action] for action in list(theta.keys())])
                p_a_s[state] = [f_a_s[state][action]/sumf_a_s if sumf_a_s != 0 else 0 for action in list(theta.keys())]

                new_state, reward, done, _ = self.env.step(selected_action)
                new_state = self.env.tau(state)[new_state]

                current_trace.append(selected_action)
                current_trace.append(reward)
                current_trace.append(new_state)
                self.traces.append(current_trace)

                return new_state, reward, done, n_a_s, f_a_s, p_a_s
            
            done = False 
            reward = 0

            while not done and reward==0:
                state, reward, done, n_a_s, f_a_s, p_a_s = pureExploration(state, n_a_s, f_a_s, p_a_s)
                
        