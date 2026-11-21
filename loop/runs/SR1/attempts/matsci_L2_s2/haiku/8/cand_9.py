import numpy as np
def equation(eps, T, params):
    a,b,c,d,e = params[:5]
    return a + b*np.log(eps) + c/eps + d*eps**3*T**2 + e*np.exp(-0.5*eps)
