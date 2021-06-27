from numpy.lib.function_base import select
from .mcts_node import MonteCarloTreeSearchNode
import numpy as np
import random
from collections import defaultdict

class MonteCarloTreeSearch():
    def __init__(self, env, iterations):
        self.env = env
        self.iterations = iterations
        self.initial_mcts_state = MonteCarloTreeSearchNode(state = self.env.specification.initial_state[:2])
        self.initial_mcts_state._untried_actions = list(self.env.specification.ACTIONS)
        self.all_actions = list(self.env.specification.ACTIONS)
        print(self.all_actions)
        self.reward_goal = 100 #self.env.specification.reward_goal
        self.total = 0
        #self.epsilon = 1


    def mcts(self, iterations):
        self.env.reset()
        if iterations>0:
            #self.epsilon = self.epsilon - (self.iterations-iterations)*0.0001
            print("iteration ", iterations)

            print("initial state: ", self.initial_mcts_state.state)

            mcts_state, done, reward, steps = self.select(self.initial_mcts_state, False, 0, 0, True)

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
                    normalized_reward = 0
                    reward, steps = self.simulate(mcts_next_state, done, reward, steps)
                    simulation_result = 1 if reward==self.reward_goal else -1
                    if simulation_result == 1:
                        normalized_reward = self.reward_goal/steps
                        self.total+=normalized_reward
                    print(simulation_result, normalized_reward)

                    self.backpropagate(mcts_next_state, simulation_result)

            print(reward)
            return self.mcts(iterations-1)

        
        else:
            print("AVERAGE REWARD:", self.total/self.iterations)
            return self.initial_mcts_state



    def select(self, mcts_state, done, reward, steps, select_flag): #la select va sempre chiamata sulla radice
        if not done and len(mcts_state.children) > 0 and reward!=self.reward_goal and select_flag:
            
            '''if best_child.q()<0:
                best_child, done, reward, steps = self.expand(mcts_state, steps) #quando espandiamo un nuovo nodo, questo nodo non avrà figli, quindi la select uscirà al ciclo successivo
                best_child._number_of_visits+=1
                best_action = best_child.parent_action
                #print(f"action:{best_action}, state: {best_child.state}")
                mcts_state = best_child
                print(f"selected state: {best_child.state}, {best_action}")
                return self.select(mcts_state, done, reward, steps+1)'''
            if len(mcts_state.children) < len(self.all_actions):
                select_flag = False
                return self.select(mcts_state, done, reward, steps, select_flag)

            else:
                best_child = self.best_child(mcts_state) 
                best_child._number_of_visits+=1
                best_action = best_child.parent_action
                print(f"selected state: {best_child.state}, {best_action}")
                state, reward, done, _ = self.env.step(best_action)
                #print(f"action:{best_action}, state: {state}")
                mcts_state = best_child
                return self.select(mcts_state, done, reward, steps+1, select_flag) 
        
        #print("selected state:", mcts_state.state)
        return mcts_state, done, reward, steps



    def expand(self, mcts_state, steps):
        '''theta = self.env.theta(mcts_state.state)
        tau = self.env.tau(mcts_state.state)
        print("tau ", tau)'''

            
        if len(mcts_state.children) < len(self.all_actions):
            selected_action = random.choice(mcts_state._untried_actions) #exploration
            mcts_state._untried_actions.remove(selected_action)
        else:
            best_child = self.best_child(mcts_state)  
            selected_action = best_child.parent_action


        state, reward, done, _ = self.env.step(selected_action)

        node = MonteCarloTreeSearchNode(state = state,
                                        parent = mcts_state, 
                                        parent_action=selected_action,
                                        _untried_actions=self.all_actions.copy())

        if node not in mcts_state.children:
            mcts_state.children.append(node)


        return node, done, reward, steps+1



    def best_child(self, state, c_param=0.1):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(c.parent.n()) / c.n())) if c.n()!=0 else 0 for c in state.children]
        return state.children[np.argmax(choices_weights)]



    def simulate(self, current_simulation_state, done, reward, steps):
        while not done and reward==0:


            action = self.simulation_policy(self.all_actions)
            print("simulated state: ", current_simulation_state.state, action)
                                                             
            state, reward, done, info = self.env.step(action)

            current_simulation_state = MonteCarloTreeSearchNode(state = state,
                                                                parent=current_simulation_state,
                                                                parent_action=action)

            #print(state, reward, info)
            steps += 1

        print("simulated state: ", current_simulation_state.state)
        #print(reward)
        return reward, steps

        


    def simulation_policy(self, possible_moves):
        print("possible moves: ", possible_moves)
        return random.choice(possible_moves)



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
                theta = self.env.theta(state_mcts.state)
                tau = self.env.tau(state_mcts.state)

                best_child = self.best_child(state_mcts)
                best_child._number_of_visits+=1
                best_action = best_child.parent_action
                state, reward, done, _ = self.env.step(best_action)
                best_child.state = tau[state]
                print(f"State: {state_mcts.state}, Action: {best_action}, Reward: {reward}")

                state_mcts = best_child
        
        print(f"State: {best_child.state}")
            
        

