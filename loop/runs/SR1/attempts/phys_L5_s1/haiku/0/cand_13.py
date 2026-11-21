import numpy as np
def equation(t, x, v, params):
    A,g,c,k,b,d,e0,s,m=params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]
    return A*np.exp(-g*t)+e0-c*v-k*x-b*x**3-d*v*np.abs(v)-s*np.sin(x)-m*v**3
