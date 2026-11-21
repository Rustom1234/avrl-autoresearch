import numpy as np
def equation(t, A, params):
    a,n,b,c,d,e,f = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return (c*(1-np.exp(-d*t)) + e)*(1 + a*A**n*np.exp(-b*t)) + f*A
