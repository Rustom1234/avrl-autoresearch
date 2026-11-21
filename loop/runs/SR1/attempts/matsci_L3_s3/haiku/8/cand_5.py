import numpy as np
def equation(eps, T, params):
    a,b,c,d,f,g,h = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return a*T**b + (c+d*T)*np.exp(-f*eps) + g*(1-np.exp(-h*eps))
