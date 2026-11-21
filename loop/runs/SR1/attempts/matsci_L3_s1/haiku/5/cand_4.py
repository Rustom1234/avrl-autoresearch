import numpy as np
def equation(eps, T, params):
    a,b,c,d,f,g,h,k=params[:8]
    A = a + b*T + c*T**2
    p = d + f*T
    q = g + h*T
    return A + k*eps**p*np.exp(-q*eps)
