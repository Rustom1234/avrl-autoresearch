import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return a*np.exp(-b*T)+c*eps**d*np.exp(e*T)+f*eps*np.exp(-g*eps)
