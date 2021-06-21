import random
import numpy as np

from numpy.random import normal

class S3M():
    def __init__(self, env):
        self.env = env
        self.initial_state = self.env.specification.initial_state
        self.traces = {} # {h: [{a: [s...]}, count]} 
        self.tr = {} # {h: {a: {s: P(s|h,a)}}}
        self.h_a_s = []
        
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
                for a in self.traces[h][0]: # Traces {h: [{a: [s...]}, count]} 
                    count_tot = self.traces[h][1]
                    for s in self.traces[h][0][a]:
                        count_obs = self.traces[h][0][a].count(s)
                        if not h in self.tr:
                            self.tr[h] = {a:{s: count_obs/count_tot}}
                        else:
                            if not a in self.tr[h]:
                                self.tr[h][a] = {s: count_obs/count_tot}
                            else:
                                self.tr[h][a][s] = count_obs/count_tot
                        if not (h,a,s) in self.h_a_s:
                            self.h_a_s.append((h,a,s))


        return 

    def merger(self,Tr,epsilon):
        '''
            Merge distributions associated to two similar traces
        '''
        # Prendere ogni coppia di h+a+s in Tr
        # Calcoliamo la Dkl e vediamo se è minore di epsilon
        # In caso affermativo mettiamo una delle due nel dizionario Tr'
        # Iteriamo su Tr' fintanto che non ci siano più cluster accorpabili

        trP = {} # Tr' {h: {a: {s: P(s|h,a)}}}

        for (h1,a1,s1) in self.h_a_s:
            for (h2,a2,s2) in self.h_a_s:
                if h1 != h2 or a1 != a2 or s1 != s2:
                    if self.traces[h1][1] >= self.traces[h2][1]: # w1 >= w2 >= min_samples
                        d_kl = – sum([x in X P(x) * log(Q(x) / P(x))])

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
        