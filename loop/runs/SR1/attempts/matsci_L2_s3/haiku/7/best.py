import numpy as np
def equation(eps, T, params):
    a,b,c,d,f,p,q,r,s = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]
    return a + b*T + c*T**p + d*T**q + (f*T**r)*eps**2 + 0*s