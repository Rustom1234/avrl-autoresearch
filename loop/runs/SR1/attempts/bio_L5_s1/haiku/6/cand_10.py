import numpy as np
def equation(t, P, params):
    r,K,H,c=params[0],params[1],params[2],params[3]
    return r*P*(1-P/K) - H*np.exp(-c*P)
