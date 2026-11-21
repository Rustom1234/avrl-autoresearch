import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g,k,m,h = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9]
    return a*np.exp(-k*eps) + b + c*eps + d*T + e*T*eps + f*T*np.exp(-m*eps) + g*eps**2 + h*T*eps**2
