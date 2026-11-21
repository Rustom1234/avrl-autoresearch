import numpy as np
def equation(eps, T, params):
    a,p,b,q,d,r = params[0],params[1],params[2],params[3],params[4],params[5]
    return a*T**p + b*T**q + d*T**r*eps**2
