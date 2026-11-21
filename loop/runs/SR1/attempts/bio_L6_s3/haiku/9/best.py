import numpy as np
def equation(t, P, params):
    a0,a1,a2,a3,a4,a5,a6,k=params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    g=np.tanh(k*t)
    return a0*g + a1*P + a2*P**2 + a3*P**3 + g*(a4*P + a5*P**2 + a6*P**3)
