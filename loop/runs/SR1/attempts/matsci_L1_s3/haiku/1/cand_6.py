import numpy as np
def equation(eps, T, params):
    a,b,c,d,k,m = params[0],params[1],params[2],params[3],params[4],params[5]
    s0 = a + b*T + c*T**2
    Q = d + k*T
    return -(s0 + Q*(1-np.exp(-m*eps)))