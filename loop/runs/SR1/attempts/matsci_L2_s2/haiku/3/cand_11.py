import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g,k,h = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]
    return a*np.exp(-k*eps)*np.cos(b*eps+c*T+d) + e + f*eps + g*T + h*T*eps
