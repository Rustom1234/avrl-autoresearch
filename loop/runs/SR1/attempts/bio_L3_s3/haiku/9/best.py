import numpy as np
def equation(t, P, params):
    a,b,c,d,e,k,A,B,w,f=params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9]
    return a + b*P + c*P**2 + (d+e*P)*np.exp(-k*t) + (A+B*P)*np.sin(w*t+f)
