import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g,h,k,m = params
    L=np.log(eps)
    return a + b*T + c*T**2 + d*eps + e*eps*T + f*L + g*T*L + h*eps**2 + k*eps**2*T + m*eps*T**2