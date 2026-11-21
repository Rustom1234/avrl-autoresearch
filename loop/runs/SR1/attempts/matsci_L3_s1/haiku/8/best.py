import numpy as np
def equation(eps, T, params):
    a0,a1,a2,a3,b0,b1,b2,c0,c1,k = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9]
    base = a0 + a1*T + a2*T**2 + a3/T
    amp = b0 + b1*T + b2/T
    return base + amp*eps*np.exp(-np.abs(k)*eps) + (c0+c1*T)*eps**2*np.exp(-np.abs(k)*eps)
