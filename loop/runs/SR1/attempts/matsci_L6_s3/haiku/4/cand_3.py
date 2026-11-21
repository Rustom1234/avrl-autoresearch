import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g,h=params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return -(a + b*T + c*T**2 + (d + e*T)*(1-np.exp(-(f+g*T)*eps)) + h*eps)