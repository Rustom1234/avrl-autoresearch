import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g,h,k,m,n = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9],params[9]
    return a + b*T + (c + d*T)*(1-np.exp(-(e+f*T)*eps)) + (g+h*T)*eps + k*T**2 + m*eps**2