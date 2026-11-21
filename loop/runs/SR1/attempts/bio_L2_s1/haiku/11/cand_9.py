import numpy as np
def equation(t, P, params):
    a,b,c,d,p,q,e=params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return a + b*t + c*P**p + d*t*P**q + e*P
