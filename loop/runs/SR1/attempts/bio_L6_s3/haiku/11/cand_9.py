import numpy as np
def equation(t, P, params):
    a,b,c,d,e,f=params[0],params[1],params[2],params[3],params[4],params[5]
    s=1-np.exp(-b*t**c)
    return -(a*s*P - d*s*P**2 + e*P**2 + f*P)