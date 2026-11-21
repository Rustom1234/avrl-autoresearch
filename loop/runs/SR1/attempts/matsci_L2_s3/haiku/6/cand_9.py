import numpy as np
def equation(eps, T, params):
    a,p,d,c = params[0],params[1],params[2],params[3]
    return a*T**p + d*np.sqrt(T)*eps**2 + c*np.sqrt(T)*eps**0
