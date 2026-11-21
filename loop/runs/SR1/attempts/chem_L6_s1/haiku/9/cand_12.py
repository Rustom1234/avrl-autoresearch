import numpy as np
def equation(t, A, params):
    a,b,c,d,e,f,g = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return (a + b*t)*A + (c + d*t) + e*np.exp(f*t)*A**g
