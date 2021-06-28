import random
import numpy as np
import ast
from math import log, log2, inf
from numpy.random import normal

class S3M():
    def __init__(self, env):
        self.env = env
        self.initial_obs = self.env.specification.initial_state[:2]
        self.traces = {} # {h: [{a: [s...]}, count]} 
        self.tr = {} # {h: {a: {s: P(s|h,a)}}}
        self.h_a = []
        self.max_dkl = 0
        self.all_actions = list(self.env.specification.ACTIONS)
        self.best_loss = float("inf")
        
    def sample(self, mode=0):
        ''' Sample a new trace by exploring the environment
        mode = 0 for pure Exploration
               1 for Smart Sampling (not yet implemented)
        '''
        self.env.reset()
        current_trace = [self.initial_obs]
        state = self.initial_obs
        n_a_s = {}
        f_a_s = {}
        p_a_s = {}

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
        for h in self.traces:   # {h: [{a: [s...]}, count]} 
            if self.traces[h][1] >= min_samples:
                for a in list(self.traces[h][0]):
                    count_tot = len(self.traces[h][0][a])
                    for obs in self.traces[h][0][a]:
                        count_obs = self.traces[h][0][a].count(obs)
                        p = count_obs/count_tot

                        if not h in self.tr:
                            self.tr[h] = {a:{obs: p}}     # tr = {h: {a: {o: p}}}
                        else:
                            if not a in self.tr[h]:
                                self.tr[h][a] = {obs: p}
                            else:
                                self.tr[h][a][obs] = p
                        
                        if not (h,a) in self.h_a:
                            self.h_a.append((h,a))

                        
                        
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

    def merger(self,epsilon):
        '''
            Merge distributions associated to two similar traces
        '''
        # Prendere ogni coppia di h+a+s in Tr = {h: {a: {o: p(o|h,a)}}}
        # Calcoliamo la Dkl e vediamo se è minore di epsilon
        # In caso affermativo mettiamo una delle due nel dizionario Tr' (quale? Forse è uguale. Rima baciata)
        # Iteriamo su Tr' fintanto che non ci siano più cluster accorpabili
        # Tr' {h: {a: {o: P(o|h,a)}}} 
        h_aP = self.h_a.copy()
        trP = self.tr.copy()

        while len(h_aP) > 1:
            (h1,a1) = random.choice(h_aP)
            all_obs1 = list(self.tr[h1][a1])
            min_d_kl = float('inf')
            min_tuple = ()        
            for (h2, a2) in self.h_a:
                if h1 != h2 or a1 != a2:
                    all_obs2 = list(self.tr[h2][a2])              
                    if sorted(all_obs1) == sorted(all_obs2):
                        # print("SOPRA\nTrP1")
                        # print(trP[h1][a1])
                        # print("TrP2")
                        # print(trP[h2][a2])
                        # print(all_obs1)
                        # print(all_obs2)
                        prob_seq1 = [self.tr[h1][a1][obs] for obs in all_obs1]
                        prob_seq2 = [self.tr[h2][a2][obs] for obs in all_obs2]
                    
                        if self.traces[h1][1] >= self.traces[h2][1]: # w1 >= w2 >= min_samples
                            d_kl = self.kl_divergence(prob_seq1, prob_seq2)
                        else:
                            d_kl = self.kl_divergence(prob_seq2, prob_seq1)
                        # print(f"Siamo belli e questa è la Dkl: {d_kl}\n\n")
                        if d_kl < epsilon:
                            if d_kl < min_d_kl:
                                min_tuple = (h2,a2)
                                min_d_kl = d_kl
            if min_tuple != ():
                # print(min_tuple)
                # print(f"({h1},{a1})")
                # print("TrP1")
                # print(trP[h1][a1])
                # print("TrP2")
                # print(trP[min_tuple[0]][min_tuple[1]])
                w1 = self.traces[h1][1]
                w2 = self.traces[min_tuple[0]][1]
                w = w1+w2
                for obs in all_obs1:
                    trP[h1][a1][obs] = ( w1*self.tr[h1][a1][obs] + w2*self.tr[min_tuple[0]][min_tuple[1]][obs] ) / w  # Eq.(2)
                
                if min_tuple in h_aP:
                    h_aP.remove(min_tuple)
                self.h_a.remove(min_tuple)

                if h1 != min_tuple[0]:
                    del trP[min_tuple[0]][min_tuple[1]]
                          
            else:
                h_aP.remove((h1,a1))          
                        
        return trP

    def merge_histories(self,list_eps):
        '''
            Merge current histories by using different epsilon
        '''
        for epsilon in list_eps:
            TrP = self.merger(epsilon)
            print(f"TR {len(self.tr)}|{sum([len(list(self.tr[v])) for v in self.tr])} - TRP {len(TrP)}|{sum([len(list(TrP[v])) for v in TrP])}")
            
            loss = self.calc_loss(TrP)
            if loss < self.best_loss and loss != 0.:
                self.tr = TrP.copy() # Vedere se copy risulta essere necessario
                self.best_loss = loss
        
        return

    def calc_loss(self, TrP):
        '''
            Compute the loss by exploiting function (3)
        '''
        # Tr: {h: {a: {o: P(o|h,a)}}} 
        P_h_Tr = {}
        for h in list(TrP):
            lh = ast.literal_eval(h)
            P_h_Tr[h] = 1 # P(h | Tr )
            for i in range(3, len(lh), 3):
                h_curr = str(lh[:i-2])
                if h_curr in list(TrP) and lh[i-2] in list(TrP[h_curr]) and lh[i] in list(TrP[h_curr][lh[i-2]]):
                    P_h_Tr[h] *= TrP[h_curr][lh[i-2]][lh[i]] 
            P_h_Tr[h] = log(P_h_Tr[h])
        loss = -sum(P_h_Tr.values()) # MANCA SECONDO TERMINE
        return loss
    
    def mealy_generator(self):
        '''
            Generate the Mealy machine
        '''

        return
        