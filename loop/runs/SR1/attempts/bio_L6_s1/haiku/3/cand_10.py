import numpy as np
def equation(t, P, params):
    a,b,K,c = params[0],params[1],params[2],params[3]
    return (a + b*t + c*t**2)*P*np.log(K/P)