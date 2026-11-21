import numpy as np
def equation(eps, T, params):
    a,b,c,d,f,g,h = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    s0 = a + b*T + c*T**2
    K = d + f*T
    n = g + h*T
    return -(s0 + K*eps**n)