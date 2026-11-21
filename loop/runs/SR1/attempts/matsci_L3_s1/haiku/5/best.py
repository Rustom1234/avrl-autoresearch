import numpy as np
def equation(eps, T, params):
    a,b,c,d,f,p,q,g,h,k=params
    return a + b*T + c*T**2 + (d + f*T)*np.power(eps,p)*np.exp(-q*eps) + g*eps + h*eps**2 + k*eps*T
