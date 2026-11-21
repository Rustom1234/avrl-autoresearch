import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g,h,k,m = params
    return -(a + b*T + c*np.log(eps) + d*eps*np.exp(-e*eps) + f*T*eps + g*np.log(eps)**2 + h*eps**2)