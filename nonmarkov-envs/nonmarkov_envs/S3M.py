import random
import numpy as np
import ast
from math import log, log2, inf, sqrt
from numpy.random import normal
import copy 


class S3M():
    def __init__(self, env):
        self.env = env
        self.initial_obs = self.env.specification.initial_state[:2]
        self.traces = {} # {h: [{a: {o: count_obs}}, count]}       # OLD: {h: [{a: [o...]}, count]}
        self.tr = {} # {h: {a: cluster_idx}}                       # OLD: {h: {a: {s: P(s|h,a)}}}
        self.cl = {} # {cluster_idx: [{o: P(o|h,a)}, weight]}
        self.max_dkl = 0
        self.all_actions = list(self.env.specification.ACTIONS)
        self.best_loss = float("inf")
        self.next_cluster_idx = 0 
        
    def sample(self, mode=0):
        ''' Sample a new trace by exploring the environment
        mode = 0 for pure Exploration
               1 for Smart Sampling (not yet implemented)
        '''
        self.env.reset()
        current_trace = [self.initial_obs]
        state = self.initial_obs
        n_a_s = {} # number of times an action a is applied to a state s
        f_a_s = {} # 1 - n_a_s / (sum_a n_a_s)
        p_a_s = {} # f_a_s / sum_a f_a_s | Probability of selecting an action a from s

        if mode == 0:

            def pureExploration(state):   
                
                if n_a_s.get(state) == None:
                    n_a_s[state] = [{}, 0]  # n = {s: [{a: num}, num]}
                    f_a_s[state] = {}       # f = {s: {a: num}}
                    p_a_s[state] = []       # p = {s: [distribution]}
                    for action in self.all_actions:
                        n_a_s[state][0][action] = 0
                        f_a_s[state][action] = 0
                        p_a_s[state] = np.random.uniform(0,1,len(self.all_actions))
                
                selected_action = random.choices(self.all_actions, p_a_s[state])[0]
                n_a_s[state][0][selected_action] += 1
                n_a_s[state][1] += 1
                f_a_s[state][selected_action] = 1 - n_a_s[state][0][selected_action] / n_a_s[state][1]

                sumf_a_s = sum([f_a_s[state][action] for action in self.all_actions])
                p_a_s[state] = [f_a_s[state][action]/sumf_a_s if sumf_a_s != 0 else 0 for action in self.all_actions]
                
                new_state, reward, done, _ = self.env.step(selected_action)
                
                if self.traces.get(str(current_trace)) == None:
                    self.traces[str(current_trace)] = [{selected_action: {new_state: 1}}, 1]    # {h: [{a: {o: count_obs}}, count]}
                else:
                    if not selected_action in self.traces[str(current_trace)][0]:
                        self.traces[str(current_trace)][0][selected_action] = {new_state: 1}
                    elif not new_state in self.traces[str(current_trace)][0][selected_action]:
                        self.traces[str(current_trace)][0][selected_action][new_state] = 1
                    else:
                        self.traces[str(current_trace)][0][selected_action][new_state] += 1
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
            # count delle volte totali in cui un'azione a è stata applicata ad h
            # P(o|h,a) = count1 / count2
            # Mettere P(o|h,a) in Tr

        for h in self.traces:   # {h: [{a: {o: count_obs}}, count]}   # OLD: {h: [{a: [o...]}, count]}
            if self.traces[h][1] >= min_samples:  # BUG: DEVE ESSERE CALCOLATO SU HA!
                for a in list(self.traces[h][0]):
                    count_tot = sum(self.traces[h][0][a].values())
                    for obs in self.traces[h][0][a]:
                        count_obs = self.traces[h][0][a][obs]
                        p = count_obs/count_tot

                        if not h in self.tr:
                            self.tr[h] = {a: self.next_cluster_idx}     # tr = {h: {a: cluster_idx}}
                            self.cl[self.next_cluster_idx] = [{obs: p}, count_tot]      # cl = {cluster_idx: [{o: P(o|h,a)}, weight]}
                            self.next_cluster_idx += 1
                        else:
                            if not a in self.tr[h]:
                                self.tr[h][a] = self.next_cluster_idx
                                self.cl[self.next_cluster_idx] = [{obs: p}, count_tot]
                                self.next_cluster_idx += 1
                            else:
                                c_index = self.tr[h][a]
                                self.cl[c_index][0][obs] = p
                                self.cl[c_index][1] = count_tot  # BUG: non deve essere sovrascritto il peso, ma incrementato

                        # if not (h,a) in self.h_a:
                        #     self.h_a.append((h,a))

        return 

	
    # compute the kl divergence
    def kl_divergence(self, p, q):
        return sum(p[i] * np.log(p[i]/q[i]) if p[i] != 0 and q[i] != 0 else 0 for i in range(len(p)))

    def merger(self,epsilon):
        '''
            Merge distributions associated to two similar traces
        '''
        # Prendere ogni coppia di h+a+s in Tr = {h: {a: {o: p(o|h,a)}}}
        # Calcoliamo la Dkl e vediamo se è minore di epsilon
        # In caso affermativo mettiamo una delle due nel dizionario Tr' (quale? Forse è uguale. Rima baciata)
        # Iteriamo su Tr' fintanto che non ci siano più cluster accorpabili
        # Tr' {h: {a: {o: P(o|h,a)}}} 

        trP = copy.deepcopy(self.tr)
        clP = copy.deepcopy(self.cl)       

        cluster_indices = sorted(list(clP))
        if len(cluster_indices) > 1:
            max_idx = cluster_indices[-1]
            curr_idx = cluster_indices[0]
        else:
            max_idx = -1
            curr_idx = -1

        while curr_idx < max_idx:

            all_obs1 = list(clP[curr_idx][0])
            min_d_kl = float('inf')
            min_index = -1 

            for i in range(1,len(cluster_indices)):  # Possible bug: see Note (2)
                idx_c2 = cluster_indices[i]
                all_obs2 = list(clP[idx_c2][0])
                
                if sorted(all_obs1) == sorted(all_obs2):
                    prob_seq1 = [clP[curr_idx][0][obs] for obs in all_obs1]
                    prob_seq2 = [clP[idx_c2][0][obs] for obs in all_obs2]
                    
                    if clP[curr_idx][1] >= clP[idx_c2][1]: # w1 >= w2 >= min_samples
                        d_kl = self.kl_divergence(prob_seq1, prob_seq2)
                    else:
                        d_kl = self.kl_divergence(prob_seq2, prob_seq1)

                    if d_kl < epsilon and d_kl < min_d_kl:
                        min_index = idx_c2
                        min_d_kl = d_kl

            if min_index != -1:
            
                w1 = clP[curr_idx][1]
                w2 = clP[min_index][1]                
                w = w1+w2
                for obs in all_obs1: # {cluster_idx: [{o: P(o|h,a)}, weight]}
                    clP[curr_idx][0][obs] = ( w1*clP[curr_idx][0][obs] + w2*clP[min_index][0][obs])/w  # Eq.(2)

                for h in trP:
                    for a in trP[h]:
                        if min_index == trP[h][a]:
                            trP[h][a] = curr_idx

                del clP[min_index]
                clP[curr_idx][1] = w
                cluster_indices.remove(min_index)

            else:
                cluster_indices.remove(curr_idx)    # Può essere cambiata con curr_idx += 1 se Note (2) è corretta

            if len(cluster_indices) > 1:
                curr_idx = cluster_indices[0]
                max_idx = cluster_indices[-1]
            else:
                break         
        
        return trP, clP


    def merge_histories(self,list_eps):
        '''
            Merge current histories by using different epsilon
        '''

        self.best_loss = float("inf")

        for epsilon in list_eps:
            trP, cP = self.merger(epsilon)
                        
            if len(trP):
                loss = self.calc_loss(trP, cP)
            else:
                loss = float("inf")

            if loss < self.best_loss and loss > 0.:
                # print(f"New best loss! Old: {self.best_loss}, new: {loss}")
                self.tr = copy.deepcopy(trP)
                self.cl = copy.deepcopy(cP)

                self.best_loss = loss

        return

    def calc_loss(self, trP, cP):
        '''
            Compute the loss by exploiting function (3)
        '''
        # Tr: {h: {a: {o: P(o|h,a)}}} 
        lamba = 0.5
        P_h_Tr = {}
        num_prob = 0
        for h in list(trP):
            #print("DENTROOO")
            #print(trP)
            lh = ast.literal_eval(h)
            P_h_Tr[h] = 1 # P(h | Tr )
            for i in range(3, len(lh), 3):
                curr_history = str(lh[:i-2])
                last_action = lh[i-2]
                next_obs = lh[i]
                index_cp = trP[curr_history][last_action]
                P_h_Tr[h] *= cP[index_cp][0][next_obs]

            P_h_Tr[h] = log(P_h_Tr[h])
        
        # Manca un altro prodotto con l'ultima osservazione prodotta dall'ultima azione!

        for cluster_idx in list(cP):
            num_prob += len(cP[cluster_idx][0])
        

        # if len(trP) > 0:
        loss = - sum(P_h_Tr.values()) + lamba * log(num_prob)
        # print(- sum(P_h_Tr.values()))
        # print(log(num_prob))
        # print(f"loss: {loss}")
        # else:
        #     print("Dentro!")
        #     loss = float("inf")
        return loss
    
    def mealy_generator(self):
        '''
            Generate the Mealy machine
        '''

        return
        