import numpy as np
def equation(eps, T, params):
    A,B,T0,K,n,C,D,E = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return (A + B*(T-T0)**2) * eps**K*np.exp(-n*eps) + C + D*np.log(eps) + E*np.exp(-T)
