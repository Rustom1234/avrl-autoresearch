import numpy as np
def equation(eps, T, params):
    a,b,c,d,q,p,g,h=params[:8]
    return a + b*T + (c + d*T)*eps**p*np.exp(-q*eps) + g*eps + h*eps*T
