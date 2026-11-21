import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g,h = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return (a+c*T+d*T**2)*(1-np.exp(-b*eps))+e*eps**f*np.exp(-g*T)+h*eps*T
