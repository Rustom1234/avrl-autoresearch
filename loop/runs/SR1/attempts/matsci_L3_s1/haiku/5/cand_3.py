import numpy as np
def equation(eps, T, params):
    a,b,c,d,f,g,h,k,m,n=params
    return (a + b*T + c*T**2) + (d + e*T if False else d*T)*(1-np.exp(-f*eps)) - g*eps + h*eps*T + k*np.log(eps)*T + m*eps**2 + n*np.log(eps)
