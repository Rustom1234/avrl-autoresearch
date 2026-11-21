import numpy as np
def equation(t, P, params):
    a0,a1,K,b0,b1 = params[0],params[1],params[2],params[3],params[4]
    return (a0 + a1*t)*P*np.log(np.abs(K)/P) + (b0 + b1*t)*np.log(np.abs(K)/P)
