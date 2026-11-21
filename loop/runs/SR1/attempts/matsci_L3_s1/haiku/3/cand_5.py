import numpy as np
def equation(eps, T, params):
    A,B,T0,a,b,C = params[0],params[1],params[2],params[3],params[4],params[5]
    return A*eps**a*np.exp(-b*eps)*(1+B*(T-T0)**2) + C*T + 0*eps
