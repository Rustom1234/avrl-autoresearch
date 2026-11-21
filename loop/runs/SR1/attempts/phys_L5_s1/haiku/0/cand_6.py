import numpy as np
def equation(t, x, v, params):
    A,w,c,k,b,d,p,q=params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return A*np.cos(w*t)+p*np.sin(w*t)-c*v-k*x-b*x**3-d*v**3 + q
