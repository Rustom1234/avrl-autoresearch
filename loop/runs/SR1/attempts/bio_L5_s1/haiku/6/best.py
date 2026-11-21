import numpy as np
def equation(t, P, params):
    a,H,c=params[0],params[1],params[2]
    return a*P**2 - H/(1+c*P)
