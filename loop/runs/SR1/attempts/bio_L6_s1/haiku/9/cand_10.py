import numpy as np
def equation(t, P, params):
    a,b,c,d,e,f,g = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return a + b*t + c*np.abs(P)**d + e*t*P + f*P*t**2 + g*P**2
