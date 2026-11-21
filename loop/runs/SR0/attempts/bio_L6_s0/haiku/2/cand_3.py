import numpy as np
def equation(t, P, params):
    a,b,c,d,e=params[0],params[1],params[2],params[3],params[4]
    return (a + b*t + c*t**2)*P + d*P**2 + e*t*P**2
