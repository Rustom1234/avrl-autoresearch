import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,r = params[:6]
    return a + b*np.log(eps) + c/eps + d*eps**3*T**2 + e*np.exp(-r*eps)
