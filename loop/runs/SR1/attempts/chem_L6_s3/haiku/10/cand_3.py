import numpy as np
def equation(t, A, params):
    k1,a,n,k2,m = params[0],params[1],params[2],params[3],params[4]
    return k1*np.exp(-a*t)*A**n - k2*A**m
