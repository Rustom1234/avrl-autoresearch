import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g,h,k,m = params
    return a + b*T + c*eps + d*eps*T + e*eps**2 + f*T**2 + g*eps**2*T + h*np.log(eps) + k*np.sqrt(eps)*T + m*eps**3