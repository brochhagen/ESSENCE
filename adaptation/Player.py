'''
Subordinate file for Compression-game: Player behavior following Iterated Quantal Best Response behavior of level-1 myopic players.
'''


import numpy as np

def normalize(m): #Aux function to normalize matrix
    for i in xrange(np.shape(m)[0]):
        for j in xrange(np.shape(m)[1]):
            if m[i,j] < 0:
                m[i,j] = 0
    m = m / m.sum(axis=1)[:, np.newaxis]
    m[np.isnan(m)] = 0.
    return m

class Player: #Player class
    def __init__(self,l,c,lam,prior,priors,pr_over_pr):
        self.accumulatedRewards = {'t1': prior, 't2': 1-prior}
        self.types = ['t1','t2']
        self.signals = ['m1','m2','m3']
        self.prior = [prior,(1-prior)]
        self.inter_priors = priors
        self.pr_over_pr = pr_over_pr
        self.lam = lam
        self.lexicon = l
        self.costVector = c


    def priorAC(self): #Player prior over states
        prior = []
        sumC = sum(self.accumulatedRewards.values())
        for i in range(len(self.types)):
            prior.append(self.accumulatedRewards[self.types[i]] / sumC)
        return prior

    def sigma(self): #Level 1 sender signaling strategy
        receiverPrime = np.zeros(np.shape(np.transpose(self.lexicon)))
        for idx_pr in xrange(len(self.inter_priors)):
                receiverPrime += (self.pr_over_pr[idx_pr] *\
                                  normalize(np.transpose(self.lexicon) * self.inter_priors[idx_pr]))
        utils = np.transpose(receiverPrime)
        return normalize(np.exp(self.lam*(utils - self.costVector)))

    def rho(self): #Level 1 receiver signaling strategy
        literal_sender = normalize(self.lexicon - self.costVector)
        return normalize(np.exp(self.lam*(normalize(np.transpose(literal_sender) * self.priorAC()))))

    def selectSignal(self, givenType): #Level 1 sender selection of signal given act
        row_idx = self.types.index(givenType)
        selection_matrix = self.sigma()
        return np.random.choice(self.signals,p=selection_matrix[row_idx])

    def selectAct(self,givenSignal): #Level 1 receiver selection of act given signal
        row_idx = self.signals.index(givenSignal)
        selection_matrix = self.rho()
        return np.random.choice(self.types, p=selection_matrix[row_idx])

    def fict_receiver(self,believed_prior): #Aux_function
        return normalize(np.transpose(self.lexicon) * believed_prior)
