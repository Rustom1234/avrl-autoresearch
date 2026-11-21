import numpy as np
def equation(t, P, params):
    a,b,c,d,q,e = params[0],params[1],params[2],params[3],params[4],params[5]
    return (a + b*t + e*t**2)*P**q + c*P**2 + d
