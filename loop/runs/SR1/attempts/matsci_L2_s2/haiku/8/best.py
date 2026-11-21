import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,r,f = params[:7]
    return a + b*np.log(eps) + c/eps + f/eps**2 + d*eps**3*T**2 + e*np.exp(-r*eps)/eps
