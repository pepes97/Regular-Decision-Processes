from numpy.lib.function_base import select
from .mcts_node import MonteCarloTreeSearchNode
import numpy as np
import random
from collections import defaultdict
import warnings

class MCTS_s3m():
    def __init__(self, env, iterations, debug, step_iterations, mealy_machine, s3m):
        self.env = env
        self.iterations = iterations
        self.debug = debug
        self.map_obs = s3m.maps_obs
        self.map_actions = s3m.maps_actions
        self.mealy_machine = mealy_machine
        self.initial_mcts_state = MonteCarloTreeSearchNode(state = (self.env.specification.initial_state[:2],mealy_machine.current_node))
        self.initial_mcts_state._untried_actions = list(self.env.specification.ACTIONS)
        self.all_actions = list(self.env.specification.ACTIONS)
        self.reward_goal = 100 #self.env.specification.reward_goal
        self.total = 0
        self.step_iterations = step_iterations # 10000
        self.rewards = [0]
        #self.epsilon = 1

    def mcts(self, iterations):
        while iterations>0:
            self.env.reset()
            
            if (self.iterations - iterations) % self.step_iterations == 0 and self.iterations!=iterations:
                print(f"iteration: {self.iterations - iterations}, reward: {self.total/(self.iterations - iterations)}")
                self.rewards.append(self.total/(self.iterations - iterations))
                
            if self.debug:
                print("iteration ", iterations)

            if self.debug:
                print("initial state: ", self.initial_mcts_state.state)

            self.initial_mcts_state._number_of_visits += 1

            mcts_state, done, reward, steps = self.select(self.initial_mcts_state, False, 0, 0, True)

            # modificato reward con done
            if done or reward == self.reward_goal:
                normalized_reward = reward/steps
                self.total+=normalized_reward

                if self.debug:
                    print(f"Normalized reward: {normalized_reward}")

                if reward == self.reward_goal:
                    simulation_result = 1
                    self.backpropagate(mcts_state, simulation_result)
                else:
                    simulation_result = -1
                    self.backpropagate(mcts_state, simulation_result)

            else:
                mcts_next_state, done, reward, steps = self.expand(mcts_state, steps)
                
                if self.debug:
                    print("done: ", done, reward)
                
                # modificato reward con done
                if done or reward == self.reward_goal:
                    normalized_reward = reward/steps
                    self.total+=normalized_reward

                    if self.debug:
                        print(f"Normalized reward: {normalized_reward}")
                    if reward == self.reward_goal:
                        simulation_result = 1
                        self.backpropagate(mcts_next_state, simulation_result)
                    else:
                        simulation_result = -1
                        self.backpropagate(mcts_next_state, simulation_result)

                else:
                    normalized_reward = 0
                    reward, steps = self.simulate(mcts_next_state, done, reward, steps)
                    simulation_result = 1 if reward==self.reward_goal else -1
                    if simulation_result == 1:
                        normalized_reward = self.reward_goal/steps
                        self.total+=normalized_reward
                    
                        if self.debug:
                            print(f"Normalized reward: {normalized_reward}")

                    self.backpropagate(mcts_next_state, simulation_result)
            
            if self.debug:
                print(f"Reward: {reward}")

            if self.debug:
                a = input()
                while(a=="0"):
                    a = input()

            iterations -= 1

        print(f"iteration: {self.iterations}, reward: {self.total/(self.iterations)}")
        self.rewards.append(self.total/(self.iterations))
        
        print("AVERAGE REWARD:", self.total/self.iterations)
        return self.initial_mcts_state, self.rewards



    def select(self, mcts_state, done, reward, steps, select_flag): #la select va sempre chiamata sulla radice
        if not done and len(mcts_state.children) > 0 and reward!=self.reward_goal and select_flag:
            
            if len(mcts_state.children) < len(self.all_actions):
                select_flag = False
                return self.select(mcts_state, done, reward, steps, select_flag)

            else:
                best_child = self.best_child(mcts_state) 
                best_action = best_child.parent_action

                if self.debug:
                    print(f"selected state: {best_child.state}, selected action: {best_action}")
                
                state, reward, done, _ = self.env.step(best_action)
                
                mealy_node = self.mealy_machine.current_node
                if mealy_node != "-1":
                    mapped_action = self.map_actions[best_action]
                    mapped_state = self.map_obs[mcts_state.state[0]]
                    oa = mapped_state + mapped_action
                    try:
                        self.mealy_machine.current_node = self.mealy_machine.edges[mealy_node][oa]
                    except KeyError:
                        self.mealy_machine.current_node = "-1"


                node = MonteCarloTreeSearchNode(state = (state,self.mealy_machine.current_node),
                                        parent = mcts_state,
                                        parent_action=best_action,
                                        _untried_actions=self.all_actions.copy())

                node._number_of_visits+=1

                found = False
                for s in mcts_state.children:
                    if s == node:
                        node = s
                        found = True
                        break

                if found == False:
                    mcts_state.children.append(node)
                    
                mcts_state = node

                return self.select(mcts_state, done, reward, steps+1, select_flag) 
        
        return mcts_state, done, reward, steps



    def expand(self, mcts_state, steps):
            
        if len(mcts_state.children) < len(self.all_actions):
            selected_action = random.choice(mcts_state._untried_actions) #exploration
            mcts_state._untried_actions.remove(selected_action)
        else:
            best_child = self.best_child(mcts_state)  
            selected_action = best_child.parent_action

        state, reward, done, _ = self.env.step(selected_action)
        
        mealy_node = self.mealy_machine.current_node

        if self.debug:
            print(f"expanded state: {state}, expanded action: {selected_action}")


        if mealy_node != "-1":
            mapped_action = self.map_actions[selected_action]
            mapped_state = self.map_obs[mcts_state.state[0]]
            oa = mapped_state + mapped_action
            try:
                self.mealy_machine.current_node = self.mealy_machine.edges[mealy_node][oa]
            except KeyError:
                self.mealy_machine.current_node = "-1"


        node = MonteCarloTreeSearchNode(state = (state,self.mealy_machine.current_node),
                                        parent = mcts_state, 
                                        parent_action=selected_action,
                                        _untried_actions=self.all_actions.copy())

        if node not in mcts_state.children:
            mcts_state.children.append(node)

        node._number_of_visits+=1

        return node, done, reward, steps+1



    def best_child(self, state, c_param=0.1):
        warnings.simplefilter('ignore', RuntimeWarning) 
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(c.parent.n()) / c.n())) if c.n()!=0 else 0 for c in state.children]
        return state.children[np.argmax(choices_weights)]

        #Invece di calcolare il valore del singolo nodo, potremmo calcolare il valore dell'azione, così da selezionare quella migliore.
        #Questo perchè, scegliendo l'azione sulla base del best child, nel setting non deterministico, andiamo completamente ad ignorare 
        #il fatto che quella stessa azione potrebbe portare ad un pessimo figlio.



    def simulate(self, current_simulation_state, done, reward, steps):
        while not done and reward==0:


            action = self.simulation_policy(self.all_actions)

            if self.debug:
                print("simulated state: ", current_simulation_state.state, action)
                                                             
            state, reward, done, info = self.env.step(action)

            mealy_node = self.mealy_machine.current_node
            if mealy_node != "-1":
                mapped_action = self.map_actions[action]
                mapped_state = self.map_obs[current_simulation_state.state[0]]
                oa = mapped_state + mapped_action
                try:
                    self.mealy_machine.current_node = self.mealy_machine.edges[mealy_node][oa]
                except KeyError:
                    self.mealy_machine.current_node = "-1"

            current_simulation_state = MonteCarloTreeSearchNode(state = (state,self.mealy_machine.current_node),
                                                                parent=current_simulation_state,
                                                                parent_action=action)

            steps += 1

        if self.debug:
            print("simulated state: ", current_simulation_state.state)
        return reward, steps

    def simulation_policy(self, possible_moves):
        return random.choice(possible_moves)

    def backpropagate(self, state, result):
        state._results[result] += 1
        if state.parent:
            self.backpropagate(state.parent, result)

            
        
