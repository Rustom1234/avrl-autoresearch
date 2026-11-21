import numpy as np
def equation(t, P, params):
    a,b,c,K,th = params[0],params[1],params[2],params[3],params[4]
    r = a + b*t + c*t**2
    return P*r*(1-(P/K)**th)
