#Player behavior (literal/pragmatic sender is soft-maximizer)
import numpy as np

def normalize(m): #Auxiliary function to normalize matrices
    m = m / m.sum(axis=1)[:, np.newaxis]
    return m

class LiteralPlayer: #Literal player behavior class
    def __init__(self,lam,lexicon):
        self.lam = lam
        self.lexicon = lexicon
        self.sender_matrix = self.sender_selection_matrix(lexicon)
        self.receiver_matrix =  self.receiver_selection_matrix()
   
    def sender_selection_matrix(self,l): #Literal sender signaling behavior
        m = np.zeros(np.shape(l))
        for i in range(np.shape(l)[0]):
            for j in range(np.shape(l)[1]):
                m[i,j] = np.exp(self.lam * l[i,j])
        return normalize(m)

    def receiver_selection_matrix(self): #Literal receiver signaling behavior
        """Take transposed lexicon and normalize row-wise (prior over states plays no role as it's currently uniform)"""
        m = normalize(np.transpose(self.lexicon))
        for r in range(np.shape(m)[0]):
            if sum(m[r]) == 0:
                for c in range(np.shape(m)[1]):
                    m[r,c] = 1. / np.shape(m)[0]
        return m

class GriceanPlayer: #Level-1 myopic Gricean behavior class
    def __init__(self,alpha, lam, lexicon):
        self.alpha = alpha
        self.lam = lam
        self.lexicon = lexicon
        self.sender_matrix = self.sender_selection_matrix(lexicon)
        self.receiver_matrix = self.receiver_selection_matrix()

    def sender_selection_matrix(self,l): #Level-1 sender signaling behavior
        literalListener = normalize(np.transpose(l))
        utils = np.transpose(literalListener)
        return normalize(np.exp(self.lam*np.power(utils, self.alpha)))

    def receiver_selection_matrix(self): #Level-1 receiver signaling behavior
        """Take transposed sender matrix and normalize row-wise (prior over states plays no role as it's currently uniform)"""
        literalsender = np.zeros(np.shape(self.lexicon))
        for i in range(np.shape(self.lexicon)[0]):
            for j in range(np.shape(self.lexicon)[1]):
                literalsender[i,j] = np.exp(self.lam * self.lexicon[i,j])
        literalsender = normalize(literalsender)
        return normalize(np.transpose(literalsender))
