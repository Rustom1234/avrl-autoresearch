import numpy as np
def equation(t, A, params):
    a,b,c,d,e,g,f = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return a*A**2 + b*A + c + np.exp(f*t)*(d + e*A + g*A**2)