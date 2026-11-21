import numpy as np
def equation(t, P, params):
    a,b,c,d,e,f,g,h,k = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]
    return a + b*t + c*P + d*P**2 + e*t*P + f*t**2 + g*np.exp(-k*P) + h*t*np.exp(-k*P)
