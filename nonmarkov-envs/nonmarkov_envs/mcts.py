from numpy.lib.function_base import select
from .mcts_node import MonteCarloTreeSearchNode
import numpy as np
import random
from collections import defaultdict

class MonteCarloTreeSearch():
    def __init__(self, env, iterations):
        self.env = env
        self.iterations = iterations
        self.initial_mcts_state = MonteCarloTreeSearchNode(state = self.env.specification.initial_state)
        self.reward_goal = 100 #self.env.specification.reward_goal
        self.total = 0


    def mcts(self, iterations):
        self.env.reset()
        if iterations>0:
            print("iteration ", iterations)

            print("initial state: ", self.initial_mcts_state.state)

            mcts_state, done, reward, steps = self.select(self.initial_mcts_state, False, 0, 0)
            print("selected state: ", mcts_state.state)

            if done:
                if reward == self.reward_goal:
                    normalized_reward = reward/steps
                    self.total+=normalized_reward

            else:
                mcts_next_state, done, reward, steps = self.expand(mcts_state, steps)
                #print("done: ", done, reward)
                print("expanded state: ", mcts_next_state.state)

                if done:
                    if reward == self.reward_goal:
                        normalized_reward = reward/steps
                        self.total+=normalized_reward

                else:
                    simulation_result = 1 if self.simulate(mcts_next_state, done, reward)==self.reward_goal else -1
                    if simulation_result == 1:
                        normalized_reward = self.reward_goal/steps
                        self.total+=normalized_reward
                    print(simulation_result)

                    self.backpropagate(mcts_next_state, simulation_result)

            return self.mcts(iterations-1)

        
        else:
            print("AVERAGE REWARD:", self.total/self.iterations)
            return self.initial_mcts_state



    def select(self, mcts_state, done, reward, steps): #la select va sempre chiamata sulla radice
        while not done and len(mcts_state.children) > 0 and reward!=100:
            #theta = self.env.theta(mcts_state.state)
            #tau = self.env.tau(mcts_state.state)

            '''for action in list(theta.keys()):
                for observation in theta[action]:
                    node = MonteCarloTreeSearchNode(state = tau[observation],
                                                    parent = mcts_state, 
                                                    parent_action=action)
                    if node not in mcts_state.children:
                        mcts_state.children.append(node)'''
            
            best_child = self.best_child(mcts_state) 
            best_child._number_of_visits+=1
            best_action = best_child.parent_action
            print(f"selected state: {best_child.state}, {best_action}")
            state, reward, done, _ = self.env.step(best_action)
            #print(f"selected state: {best_child.state}, real state: {state}")
            mcts_state = best_child
            return self.select(mcts_state, done, reward, steps+1) 
        
        print("selected state:", mcts_state.state)
        return mcts_state, done, reward, steps



    def expand(self, mcts_state, steps):
        theta = self.env.theta(mcts_state.state)
        tau = self.env.tau(mcts_state.state)

        for action in list(theta.keys()):
            observation = self.select_observation(theta[action])
            node = MonteCarloTreeSearchNode(state = tau[observation],
                                            parent = mcts_state, 
                                            parent_action=action)
            if node not in mcts_state.children:
                mcts_state.children.append(node)
        
        best_child = self.best_child(mcts_state)
        best_child._number_of_visits+=1
        best_action = best_child.parent_action
        state, reward, done, _ = self.env.step(best_action)
        print(f"expanded state: {best_child.state}, real state: {state}")

        return best_child, done, reward, steps+1



    def best_child(self, state, c_param=0.1):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(c.parent.n()) / c.n())) if c.n()!=0 else 0 for c in state.children]
        return state.children[np.argmax(choices_weights)]



    def simulate(self, current_simulation_state, done, reward):
        while not done and reward==0:
            theta = self.env.theta(current_simulation_state.state)
            tau = self.env.tau(current_simulation_state.state)

            possible_moves = list(theta.keys())
            action = self.simulation_policy(possible_moves)
            print("simulated state: ", current_simulation_state.state, action)

            observation = self.select_observation(theta[action])
                                                             
            state, reward, done, info = self.env.step(action)

            #print(f"simulated state: {current_simulation_state.state}, real state: {state}")

            current_simulation_state = MonteCarloTreeSearchNode(state = tau[observation],
                                                                parent=current_simulation_state,
                                                                parent_action=action)

            #print(state, reward, info)

        print("simulated state: ", current_simulation_state.state)
        #print(reward)
        return reward



    def select_observation(self, observations):
        weights = []
        for k in observations:
            for k2 in observations[k]:
                weights.append(observations[k][k2])
        
        observation = random.choices(list(observations.keys()), weights, k=2)
        return observation[0]
        


    def simulation_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]



    def backpropagate(self, state, result):
        state._results[result] += 1
        if state.parent:
            self.backpropagate(state.parent, result)



    def print_best_path(self, state_mcts, done):
        self.env.reset()
        
        while not done:
            #print("current_state : ", state_mcts.state)
            '''for child in state_mcts.children:
                print(child.state, child.n(), child.q())'''
            if len(state_mcts.children) > 0:
                best_child = self.best_child(state_mcts)
                best_child._number_of_visits+=1
                best_action = best_child.parent_action
                state, reward, done, _ = self.env.step(best_action)
                print(f"State: {state_mcts.state}, Action: {best_action}, Reward: {reward}")

                state_mcts = best_child
        
        print(f"State: {best_child.state}")
            
        

