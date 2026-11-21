import numpy as np
def equation(t, P, params):
    a,b,c,d,e,g,h,k = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return a + b*t**g + c*np.abs(P)**h + d*t*P + e*P**2 + k*t*P**2
