import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,p,q,g=params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]
    return a + b*T**p + c*eps**q + d*eps**q*T**p + e*eps*T**p + f*eps*T + g*eps**2*T**p