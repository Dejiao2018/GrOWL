import sys
import numpy as np
from scipy.linalg import orth
from numpy.random import randn, permutation, rand
from numpy.linalg import norm
from numpy import median
from sklearn.preprocessing import normalize
from math import sqrt,exp,atan,cos,sin,pi,ceil
import ctest  
import time

def proxOWL(z, mu):
    """

    Args: 
    z:  z = x_t - lr * Gradient (f(x_t)) with lr being the learning rate

    mu:  mu = \rho_t * w where w are the OWL params. It must be nonnegative 
              and in non-increasing order. 
    """
    
    #Restore the signs of z
    sgn = np.sign(z)
    #Sort z to non-increasing order
    z = abs(z)
    indx = z.argsort()
    indx = indx[::-1]
    z = z[indx]
    
    # find the index of the last positive entry in vector z - mu  
    flag = 0
    n = z.size
    x = np.zeros((n,))
    diff = z - mu 
    diff = diff[::-1]
    indc = np.argmax(diff>0)
    flag = diff[indc]

    #Apply prox on non-negative subsequence of z - mu
    if flag > 0:
        k = n - indc
        v1 = z[:k]
        v2 = mu[:k]
        v = ctest.proxOWL_part(v1,v2)
        x[indx[:k]] = v
    else:
        pass

    # restore signs
    x = np.multiply(sgn,x)

    return x