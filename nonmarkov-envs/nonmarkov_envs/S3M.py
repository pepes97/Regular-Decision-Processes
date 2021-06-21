import random
import numpy as np
import ast
from math import log2
from numpy.random import normal

class S3M():
    def __init__(self, env):
        self.env = env
        self.initial_state = self.env.specification.initial_state
        self.traces = {} # {h: [{a: [s...]}, count]} 
        self.tr = {} # {h: {a: {s: P(s|h,a)}}}
        self.h_a_s = []
        self.h_a_sP = []
        self.max_dkl = 0
        
    def sample(self, mode=0):
        ''' Sample a new trace by exploring the environment
        mode = 0 for pure Exploration
               1 for Smart Sampling (not yet implemented)
        '''
        self.env.reset()
        current_trace = [self.initial_state]
        state = self.initial_state
        n_a_s = {}
        f_a_s = {}
        p_a_s = {}

        if mode == 0:
            def pureExploration(state):     # Marchio di fabbrica di Sveva       
                
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
                
                if self.traces.get(str(current_trace)) == None:
                    self.traces[str(current_trace)] = [{selected_action: [new_state]}, 1]
                    
                #if not selected_action in self.traces[current_trace][0]:   
                else:
                    if not selected_action in self.traces[str(current_trace)][0]:
                        self.traces[str(current_trace)][0][selected_action] = [new_state]
                    else:
                        self.traces[str(current_trace)][0][selected_action].append(new_state)
                    self.traces[str(current_trace)][1] += 1

                 
                current_trace.append(selected_action)
                current_trace.append(reward)
                current_trace.append(new_state)                

                return new_state, reward, done
            
            done = False 
            reward = 0

            while not done and reward==0:
                state, reward, done = pureExploration(state)
                

    def base_distribution(self, min_samples):
        '''
            Compute P(o|h,a)
        '''
        # Selezionare le storie che compaiono almeno min_samples
        # Per ogni storia h, per ogni azione a e per ogni stato ottenuto applicando a ad h, facciamo:
            # count di quante volte un'osservazione compare dopo applicato a ad h
            # count delle volte totali in cui un'azione è stata applicata ad h
            # P(o|h,a) = count1 / count2
            # Mettere P(o|h,a) in Tr
        for h in self.traces:
            if self.traces[h][1] >= min_samples:
                for a in self.env.specification.ACTIONS:
                    count_tot = self.traces[h][1]
                    last_s = ast.literal_eval(h)[-1]
                    theta = self.env.theta(last_s)
                    tau = self.env.tau(last_s)
                    for s in theta[a]:
                        full_s = tau[s]
                        if a in self.traces[h][0] and full_s in self.traces[h][0][a]:
                            count_obs = self.traces[h][0][a].count(full_s)
                            v = count_obs/count_tot + 0.5
                        else:
                            v = 0.5

                        if not h in self.tr:
                            self.tr[h] = {a:{full_s: v}}
                        else:
                            if not a in self.tr[h]:
                                self.tr[h][a] = {full_s: v}
                            else:
                                self.tr[h][a][full_s] = v
                        
                        if not (h,a,full_s) in self.h_a_s:
                            self.h_a_s.append((h,a,full_s))
                        
                # for a in self.traces[h][0]: # Traces {h: [{a: [s...]}, count]} 
                #     count_tot = self.traces[h][1]
                #     for s in self.traces[h][0][a]:
                #         count_obs = self.traces[h][0][a].count(s)
                #         if not h in self.tr:
                #             self.tr[h] = {a:{s: count_obs/count_tot}}
                #         else:
                #             if not a in self.tr[h]:
                #                 self.tr[h][a] = {s: count_obs/count_tot}
                #             else:
                #                 self.tr[h][a][s] = count_obs/count_tot
                #         if not (h,a,s) in self.h_a_s:
                #             self.h_a_s.append((h,a,s))


        return 

	
    # calculate the kl divergence
    def kl_divergence(self, p, q):
        return round(sum(p[i] * np.log(p[i]/q[i]) if p[i] != 0 and q[i] != 0 else 0 for i in range(len(p))), 4)

    def merger(self,epsilon,merge,h_a_s):
        '''
            Merge distributions associated to two similar traces
        '''
        # Prendere ogni coppia di h+a+s in Tr
        # Calcoliamo la Dkl e vediamo se è minore di epsilon
        # In caso affermativo mettiamo una delle due nel dizionario Tr' (quale? Forse è uguale. Rima baciata)
        # Iteriamo su Tr' fintanto che non ci siano più cluster accorpabili
        # Tr' {h: {a: {s: P(s|h,a)}}}
        h_a_sP = h_a_s.copy()
        if merge == 0: 
            return 
        else: 
            for (h1,a1,s1) in self.h_a_s:
                lh1 = ast.literal_eval(h1)
                state_seq1 = [lh1[i] for i in range(0, len(lh1), 2)]
                min_d_kl = float('inf')
                min_tuple = ()
                for (h2,a2,s2) in self.h_a_s:
                    if (h1,a1,s1) != (h2,a2,s2):
                        lh2 = ast.literal_eval(h2)
                        state_seq2 = [lh2[i] for i in range(0, len(lh2), 2)]
                        if state_seq1 == state_seq2: # {h: {a: {s: P(s|h,a)}}}
                            print(self.tr[h1][a1].values())
                            prob_seq1 = [self.tr[h1][a1][s]/sum(self.tr[h1][a1].values()) for s in self.tr[h1][a1]]
                            prob_seq2 = [self.tr[h2][a2][s]/sum(self.tr[h2][a2].values()) for s in self.tr[h2][a2]]
                            print(lh1)
                            print(lh2)
                            print(prob_seq1)
                            print(prob_seq2)
                            if self.traces[h1][1] >= self.traces[h2][1]: # w1 >= w2 >= min_samples
                                d_kl = self.kl_divergence(prob_seq1, prob_seq2)
                            else:
                                d_kl = self.kl_divergence(prob_seq2, prob_seq1)
                            print(f"Siamo belli e questa è la Dkl: {d_kl}\n\n")
                            if d_kl < epsilon:
                                if d_kl < min_d_kl:
                                    min_tuple = (h2,a2,s2)
                h_a_sP.remove(min_tuple)
                
                del self.tr[h2]
                

            return             
                        


        return

    def merge_histories(self):
        '''
            Merge current histories by using different epsilon
        '''

        return

    def calc_loss(self, TrP):
        '''
            Compute the loss by exploiting function (3)
        '''
        return
    
    def mealy_generator(self):
        '''
            Generate the Mealy machine
        '''

        return
        