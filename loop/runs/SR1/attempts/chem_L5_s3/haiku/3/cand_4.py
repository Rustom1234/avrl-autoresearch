import numpy as np
def equation(t, A, params):
    k,n,a,b,c = params[0],params[1],params[2],params[3],params[4]
    return k*A**n*t**a*np.exp(-b*t/(A+c))
