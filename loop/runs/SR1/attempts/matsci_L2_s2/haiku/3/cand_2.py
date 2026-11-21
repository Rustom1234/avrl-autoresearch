import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f = params[0],params[1],params[2],params[3],params[4],params[5]
    return a*np.cos(b*eps+c*T+d) + e*eps + f*T
