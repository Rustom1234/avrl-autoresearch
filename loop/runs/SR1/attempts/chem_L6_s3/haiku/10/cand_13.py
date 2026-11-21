import numpy as np
def equation(t, A, params):
    a,b,c,d,e,g,h,f,q = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]
    return a + b/t + c*np.exp(-d*t) + e*np.exp(-g*t) - h*t**f*np.abs(A)**q
