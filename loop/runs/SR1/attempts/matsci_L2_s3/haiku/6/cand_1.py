import numpy as np
def equation(eps, T, params):
    a,b,c,d,e = params[0],params[1],params[2],params[3],params[4]
    return a + b*T**d*eps**c + e*T
