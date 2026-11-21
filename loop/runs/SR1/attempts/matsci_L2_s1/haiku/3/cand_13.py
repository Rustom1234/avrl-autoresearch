import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,k=params[:7]
    return a + b*T + c*np.exp(-d*T) + eps*(e + f*T) + k*eps**2
