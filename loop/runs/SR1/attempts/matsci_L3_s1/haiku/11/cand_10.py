import numpy as np
def equation(eps, T, params):
    a,n,k,m,s0,s1,s2 = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return a*eps**n*np.exp(-k*eps*T**m) + s0 + s1*T + s2*T**2