import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,p,f,g,h=params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]
    return a + b*T + c/T + (d + e/T + g*T)*eps**(p+h/T) + f*eps*T