import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g,h,k,m = params
    return -((a + b*T + c*T**2)*eps**d*np.exp(-e*eps) + f + g*eps + h*T*eps)