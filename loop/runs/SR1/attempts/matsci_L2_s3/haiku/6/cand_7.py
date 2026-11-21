import numpy as np
def equation(eps, T, params):
    a,p,b,q,c,d,r,e = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return a*T**p + b*T**q*eps**c + d*T**r*eps**e
