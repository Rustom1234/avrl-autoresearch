import numpy as np
def equation(t, P, params):
    a,b,c,d,e,f=params[0],params[1],params[2],params[3],params[4],params[5]
    return P*(a + b*np.exp(-c*t) + d*P + e*P*np.exp(-c*t) + f*P**2)