import numpy as np
def equation(eps, T, params):
    a,b,c,d,f,g = params[0],params[1],params[2],params[3],params[4],params[5]
    L = np.log(T)
    return a + b*L + c*L**2 + d*eps**f + g*L*eps**f
