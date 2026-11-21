import numpy as np
def equation(t, P, params):
    a,K,h,k,c0,s=params[0],params[1],params[2],params[3],params[4],params[5]
    return a*P*(1-P/K) - h*(1-np.exp(-k*t)) + c0*P*np.exp(-s*t)
