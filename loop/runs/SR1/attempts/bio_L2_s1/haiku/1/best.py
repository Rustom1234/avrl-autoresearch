import numpy as np
def equation(t, P, params):
    c0,c1,c2,c3,c4,c5,c6,c7=params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return c0 + c1*t + c2*P + c3*P**2 + c4*P**3 + c5*t*P + c6*t*P**2 + c7*t**2
