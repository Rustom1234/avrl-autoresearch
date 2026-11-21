import numpy as np
def equation(t, A, params):
    a,b,c,e,g,f = params[0],params[1],params[2],params[3],params[4],params[5]
    return a*A**2 + b*A + np.exp(f*t)*(c + e*A + g*A**2)