import numpy as np
def equation(eps, T, params):
    a,n,b,c,d,e=params[0],params[1],params[2],params[3],params[4],params[5]
    return -(a*eps**n*np.exp(b/T) + c*np.sqrt(eps)*np.exp(d/T)*T**e)