import numpy as np
def equation(eps, T, params):
    a,b,c,d,f = params[0],params[1],params[2],params[3],params[4]
    return a*T**d*eps**(b+c*T) + f*T**2
