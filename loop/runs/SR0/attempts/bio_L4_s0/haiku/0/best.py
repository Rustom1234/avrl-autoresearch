import numpy as np
def equation(t, P, params):
    a,b,c,d,e,k = params[0],params[1],params[2],params[3],params[4],params[5]
    return a + b*t + c*t**2 + d*np.sqrt(P) + e*t*P + k*P*np.exp(-k*0 - params[6]*t)
