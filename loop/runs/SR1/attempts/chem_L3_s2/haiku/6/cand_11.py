import numpy as np
def equation(t, A, params):
    a,b,c,d,e,f,n = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return a*(1-np.exp(-c*t)) + e + b*A**n*(1-np.exp(-f*t))*np.exp(-d*t)
