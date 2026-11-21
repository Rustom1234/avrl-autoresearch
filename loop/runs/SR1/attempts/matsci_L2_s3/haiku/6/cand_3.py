import numpy as np
def equation(eps, T, params):
    a,p,b,q,c,r = params[0],params[1],params[2],params[3],params[4],params[5]
    return a*T**p + b*T**q*eps**c + 0*r
