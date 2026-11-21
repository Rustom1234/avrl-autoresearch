import numpy as np
def equation(t, P, params):
    a,b,c,d = params[0],params[1],params[2],params[3]
    return a + b*P + c*P**2 + d/P
