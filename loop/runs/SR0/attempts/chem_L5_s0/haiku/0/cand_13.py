import numpy as np
def equation(t, A, params):
    a0,a1,a2,a3,K,b1 = params[0],params[1],params[2],params[3],params[4],params[5]
    return a0 + a1*A + a2*A**2 + a3*A**2/(K**2+A**2) + b1*t*A
