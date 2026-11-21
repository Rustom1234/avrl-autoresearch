import numpy as np
def equation(eps, T, params):
    a,b,c,d,f,g,h = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return a*T**b*(1+c*np.exp(-f*eps)) + d*eps**0.5*T**0 + g*eps*T*0 + h*0
