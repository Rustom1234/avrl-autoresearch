import numpy as np
def equation(eps, T, params):
    a,b,c,k,d,e,f = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return (a + b*eps + c*eps**2)*np.exp(-k*eps)*(1 + d*T + e*T**2) + f*T