import numpy as np
def equation(eps, T, params):
    a,b,c,d,f,p,q,r,s=params
    return a + b*T + c*T**2 + (d + f*T)*np.power(eps,p)*np.exp(-(q + r*T)*eps) + s*eps*T
