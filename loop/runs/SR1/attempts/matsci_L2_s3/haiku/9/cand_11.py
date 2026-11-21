import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,p,q,f=params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return a + b*T**p + (c + d*T)*eps**q + e*eps*T + f*eps**2*T