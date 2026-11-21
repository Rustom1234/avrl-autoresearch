import numpy as np
def equation(t, P, params):
    a,b,c,d,e,k,n = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return a + b*P + c*P**2 + d*P**3 + e*np.exp(k*t)*P**n