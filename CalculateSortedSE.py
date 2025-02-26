

import numpy as np
def sorted_SE(SE):
    
    K = SE.shape[0]
    nbrOfSetups = SE.shape[1]
    A=np.reshape(SE[:,0:nbrOfSetups],(K*nbrOfSetups,1))
    sorted_SE = A[A[:,0].argsort(kind='mergesort')]
    
    return (sorted_SE)