import numpy as np
def equation(eps, T, params):
    a,b,c,d,k = params[0],params[1],params[2],params[3],params[4]
    return a + b*T**2 + c/T + d*T**1.5 + k*eps**3/T
