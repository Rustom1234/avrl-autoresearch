import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g,h,k,p = params
    return a + b*np.log(eps) + c*eps + d*eps**2 + e*T + f*T**2 + g*eps*T + h*eps**3 + k*np.log(eps)*T
