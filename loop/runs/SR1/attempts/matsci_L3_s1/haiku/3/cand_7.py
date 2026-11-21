import numpy as np
def equation(eps, T, params):
    A,B,T0,K,n,C,D = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return (A + B*(T-T0)**2 + D*T) * (1-np.exp(-K*eps))*np.exp(-n*eps) + C
