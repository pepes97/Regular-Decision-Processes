import random
import numpy as np
import ast
from math import log
import copy 
import itertools
from .utils import flexfringe
from .mealy_machine import MM

class S3M():
    def __init__(self, env):
        self.env = env
        self.initial_obs = self.env.specification.initial_state
        self.traces = {} # {ha: [{o: count_obs}, count_ha]}       
        self.tr = {} # {ha: cluster_idx}}                      
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
                
                current_trace.append(selected_action)
                if self.traces.get(str(current_trace)) == None:
                    self.traces[str(current_trace)] = [{new_state: 1}, 1]    # {ha: [{o: count_obs}, count_ha]}
                else:
                    if not new_state in self.traces[str(current_trace)][0]:
                        self.traces[str(current_trace)][0][new_state] = 1                        
                    else:
                        self.traces[str(current_trace)][0][new_state] += 1
                    self.traces[str(current_trace)][1] += 1

                current_trace.append(new_state)  
                
                return new_state, reward, done
            
            done = False 
            reward = 0

            while not done and reward==0:
                state, reward, done = pureExploration(state)
                

    def base_distribution(self):
        '''
            Compute P(o|h,a)
        '''
        # Selezionare le storie che compaiono almeno min_samples
        # Per ogni storia h, per ogni azione a e per ogni stato ottenuto applicando a ad h, facciamo:
            # count di quante volte un'osservazione compare dopo applicato a ad h
            # count delle volte totali in cui un'azione a è stata applicata ad h
            # P(o|h,a) = count1 / count2
            # Mettere P(o|h,a) in Tr
        
        for c in self.cl:
            self.cl[c][1] = 0

        
        for ha in self.traces:   # {ha: [{o: count_obs}, count_ha]}   
            #if self.traces[ha][1] >= min_samples:  # BUG: DEVE ESSERE CALCOLATO SU HA!
            count_tot = self.traces[ha][1]
            for obs in self.traces[ha][0]:
                count_obs = self.traces[ha][0][obs]
                p = count_obs/count_tot

                if not ha in self.tr:
                    self.tr[ha] = self.next_cluster_idx     # tr = {ha: cluster_idx}
                    self.cl[self.next_cluster_idx] = [{obs: p}, count_obs]      # cl = {cluster_idx: [{o: P(o|h,a)}, weight]}
                    self.next_cluster_idx += 1
                    
                else:
                    c_index = self.tr[ha]
                    self.cl[c_index][0][obs] = p
                    self.cl[c_index][1] +=count_obs

        return

	
    # compute the kl divergence
    def kl_divergence(self, p, q):
        return sum(p[i] * np.log(p[i]/q[i]) if p[i] != 0 and q[i] != 0 else 0 for i in range(len(p)))

    def merger(self,epsilon, min_samples):
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

        del_clusters = []

        for c1 in clP:
            all_obs1 = list(clP[c1][0])
            min_d_kl = float('inf')
            d_kl = float('inf')
            min_index = -1
            if c1 in del_clusters:
                continue
            

            for c2 in clP:  # Possible bug: see Note (2)
                all_obs2 = list(clP[c2][0])
                if c2 == c1 or c2 in del_clusters:
                    continue
                
                if sorted(all_obs1) == sorted(all_obs2):
                    prob_seq1 = [clP[c1][0][obs] for obs in all_obs1]
                    prob_seq2 = [clP[c2][0][obs] for obs in all_obs2]
                    
                    if clP[c1][1] >= clP[c2][1] and clP[c2][1]>=min_samples: # w1 >= w2 >= min_samples
                        d_kl = self.kl_divergence(prob_seq1, prob_seq2)

                    elif clP[c1][1] < clP[c2][1] and clP[c1][1]>=min_samples: # w1 < w2 >= min_samples
                        d_kl = self.kl_divergence(prob_seq2, prob_seq1)

                    if d_kl < epsilon and d_kl < min_d_kl:
                        min_index = c2
                        min_d_kl = d_kl

            if min_index != -1:
            
                w1 = clP[c1][1]
                w2 = clP[min_index][1]                
                w = w1+w2
                for obs in all_obs1: # {cluster_idx: [{o: P(o|h,a)}, weight]}
                    clP[c1][0][obs] = ( w1*clP[c1][0][obs] + w2*clP[min_index][0][obs])/w  # Eq.(2)

                for ha in trP:
                    if min_index == trP[ha]:
                        trP[ha] = c1

                del_clusters.append(min_index)
                clP[c1][1] = w

        for id_cluster in del_clusters:
            del clP[id_cluster]    
        
        del_clusters = []
        
        # secondo for
        for c1 in clP:
            if clP[c1][1] >= min_samples or c1 in del_clusters:
                continue
            all_obs1 = list(clP[c1][0])
            min_d_kl = float('inf')
            d_kl = float('inf')
            min_index = -1
                    
            for c2 in clP:  # Possible bug: see Note (2)
                if c2 == c1 or clP[c2][1] < min_samples or c2 in del_clusters:
                    continue
                all_obs2 = list(clP[c2][0])
                if sorted(all_obs1) == sorted(all_obs2):
                    prob_seq1 = [clP[c1][0][obs] for obs in all_obs1]
                    prob_seq2 = [clP[c2][0][obs] for obs in all_obs2]
                    
                    d_kl = self.kl_divergence(prob_seq1, prob_seq2)

                    if d_kl < min_d_kl:
                        min_index = c2
                        min_d_kl = d_kl

            if min_index != -1:    
                w1 = clP[c1][1]
                w2 = clP[min_index][1]                
                w = w1+w2
                for obs in all_obs1: # {cluster_idx: [{o: P(o|h,a)}, weight]}
                    clP[min_index][0][obs] = ( w1*clP[c1][0][obs] + w2*clP[min_index][0][obs])/w  # Eq.(2)

                for ha in trP:
                    if c1 == trP[ha]:
                        trP[ha] = min_index

                del_clusters.append(c1)
                clP[min_index][1] = w

        for id_cluster in del_clusters:
            del clP[id_cluster]     
        
        return trP, clP

    def merge_histories(self,list_eps, min_samples):
        '''
            Merge current histories by using different epsilon
        '''

        self.best_loss = float("inf")

        for epsilon in list_eps:
            trP, cP = self.merger(epsilon, min_samples)
            
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
        # Tr: {ha: c}
        # CL: {c : [{o:P}, w]} 
        lamba = 0.5
        P_h_Tr = {}
        num_prob = 0
        for ha in list(trP):
            lha = ast.literal_eval(ha)
            P_h_Tr[ha] = 1 # P(h | Tr )
            for i in range(2, len(lha), 2):
                curr_history = str(lha[:i])
                next_obs = lha[i]
                index_cp = trP[curr_history]
                P_h_Tr[ha] *= cP[index_cp][0][next_obs]

            P_h_Tr[ha] = log(P_h_Tr[ha])
        
        # Manca un altro prodotto con l'ultima osservazione prodotta dall'ultima azione!

        for cluster_idx in list(cP):
            num_prob += len(cP[cluster_idx][0])
        
        loss = - sum(P_h_Tr.values()) + lamba * log(num_prob)
        
        return loss
    
    def mealy_file_generator(self):
        '''
            Generate the file for Mealy machine
        '''
        # La struttura del file deve essere

        # sample_number = len(trP)
        # alphabet_size = prodotto cartesiamo azioni e osservazioni -> len(a) + len(o)
        # attributes = senza niente

        # Ogni riga sarà del formato seguente, per esempio:
        # (0,0), 1, (1,0), 2, (1,1), 3
        # o1 = (0,0), a1 = 1, o2=(1,0), a2 = 2 ecc
        # 1 3 o1a1/0 o2a2/1 o3a3/2
        # dove 1 iniziale viene messo perché "la storia deve essere accettata"
        # 3 è la lunghezza della storia corrente e 
        # poi, o1a1 corrispondono a osservazione+azione e 0 indica il cluster associato a quella 
        # sottostoria, quindi 0 è il cluster della sottostoria (0,0). Invece 1 è il cluster della 
        # sottostoria (0,0),1,(1,0),2 ecc.

        
        sample_number = len(self.tr)

        self.maps_actions = {}
        num_actions = 1
        list_actions = [str(a) for a in self.all_actions]

        for a in self.all_actions:
            self.maps_actions[a] = "a"+str(num_actions)
            num_actions+=1

        list_obs = []
        self.maps_obs = {}
        num_obs = 1

        for c in self.cl:
            obs_cluster = list(self.cl[c][0].keys())
            for o in obs_cluster:
                if not o in list_obs:
                    list_obs.append(o)
                    self.maps_obs[o] = "o"+str(num_obs)
                    num_obs+=1
        alphabet_size = len([e for e in itertools.product(list_obs,list_actions)])

        if sample_number !=0:
            name = "prova.txt.dat"
            with open(name, "w") as f:
                f.write(f"{sample_number} {alphabet_size} \n")
                for ha in self.tr:
                    lha = ast.literal_eval(ha)
                    f.write(f"1 {len(lha)//2} ")
                    for i in range(2,len(lha)+1,2):
                        obs = lha[i-2]
                        action = lha[i-1]
                        index_cluster = self.tr[str(lha[:i])]
                        f.write(f"{self.maps_obs[obs]}{self.maps_actions[action]}/{index_cluster} ")
                    f.write("\n")

            f.close()
        else:
            name = ""
        return name
    
    def mealy_machine(self, name):
        data = flexfringe(name, ini="../../flexfringe/dfasat/ini/batch-mealy.ini")

        mealy_machine = MM(f"{name}.ff.final.dot.json")
        mealy_machine.build_mealy()

        return mealy_machine, data

    