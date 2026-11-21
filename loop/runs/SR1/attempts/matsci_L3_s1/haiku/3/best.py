import numpy as np
def equation(eps, T, params):
    a,b,c,p,q,d,f,g,h,k = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9]
    return (a + b*(T-c)**2 + h*T**3) * eps**p/(1+q*eps**2) + d + f*np.sqrt(eps) + g*(T-c)**2 + k*eps*(T-c)**2
