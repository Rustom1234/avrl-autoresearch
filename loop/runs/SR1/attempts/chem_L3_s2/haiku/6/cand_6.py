import numpy as np
def equation(t, A, params):
    a,n,b,c,d,e = params[0],params[1],params[2],params[3],params[4],params[5]
    return a*A**n/(1+b*t) + c*t/(d+t) + e
