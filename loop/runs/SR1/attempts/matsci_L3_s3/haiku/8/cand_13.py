import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g,h,k = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]
    return a + b*T + c*T**2 + d*T**3 + (e + f*T + g*T**2)*np.exp(-h*eps) + (k + params[9]*T)*np.log(eps)
