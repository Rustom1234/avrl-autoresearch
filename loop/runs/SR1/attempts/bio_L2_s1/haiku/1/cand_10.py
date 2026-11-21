import numpy as np
def equation(t, P, params):
    a,b,c,K,n,d=params[0],params[1],params[2],params[3],params[4],params[5]
    return (a + b*t)*(1 - np.abs(P/K)**n) + c*t*P + d*P
