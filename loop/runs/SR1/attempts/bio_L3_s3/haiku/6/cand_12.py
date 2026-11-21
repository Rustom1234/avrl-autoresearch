import numpy as np
def equation(t, P, params):
    a,b,c,A,w,ph,B,d = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return a + b*P + c*P**2 + (A + B*P)*np.sin(w*t + ph) + d*t
