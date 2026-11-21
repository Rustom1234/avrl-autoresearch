import numpy as np
def equation(t, A, params):
    a,b,c,d,e,f,n,g = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return a - b*np.exp(-c*t) + A**n*(d*np.exp(-e*t) + f*np.exp(-g*t))
