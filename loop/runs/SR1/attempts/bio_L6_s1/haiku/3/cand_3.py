import numpy as np
def equation(t, P, params):
    a,b,k,c,d = params[0],params[1],params[2],params[3],params[4]
    return P*(a + b*np.exp(-k*t)) - c*P**2 + d*np.exp(-k*t)*P**2