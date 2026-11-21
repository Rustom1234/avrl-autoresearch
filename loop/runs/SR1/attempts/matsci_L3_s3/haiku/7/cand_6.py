import numpy as np
def equation(eps, T, params):
    a,b,c,n,d,f,g,h = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return a + b*T + c*eps**n*(1+d*T) + f*eps*T + g*eps*T**2 + h*T**2
