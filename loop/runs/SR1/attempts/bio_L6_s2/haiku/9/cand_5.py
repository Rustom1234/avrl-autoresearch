import numpy as np
def equation(t, P, params):
    a,b,c,p,q=params[0],params[1],params[2],params[3],params[4]
    return -P**p*(a + b*np.exp(-c*t)) + q*P
