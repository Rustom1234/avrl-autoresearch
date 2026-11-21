import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g = params[:7]
    return a + b*T + (c + e*T)*(1-np.exp(-d*eps)) + g*np.log(eps)*T + f*np.log(eps)