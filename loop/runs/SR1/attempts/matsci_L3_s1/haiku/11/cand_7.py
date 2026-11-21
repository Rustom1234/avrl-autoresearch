import numpy as np
def equation(eps, T, params):
    a0,a1,n0,n1,k0,k1,c0,c1 = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return (a0+a1*T)*eps**(n0+n1*T)*np.exp(-(k0+k1*T)*eps) + c0 + c1*T