import numpy as np
def equation(eps, T, params):
    a,p,b,q,c,d = params[0],params[1],params[2],params[3],params[4],params[5]
    return a*eps**p/(1+b*eps)**q + c*np.log(1+d*eps)
