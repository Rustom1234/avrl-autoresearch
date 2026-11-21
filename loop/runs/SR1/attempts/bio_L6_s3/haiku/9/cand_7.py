import numpy as np
def equation(t, P, params):
    a0,a1,a2,b0,b1,k,a3,b2=params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    g=1-np.exp(-k*t)
    return a0+a1*P+a2*P**2+a3*P**3+(b0+b1*P+b2*P**2)*g
