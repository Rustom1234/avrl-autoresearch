import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g=params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return a + b*T + c*eps + d*eps**2 + e*eps*T + f*eps**2*T + g*T**2*eps