import numpy as np
def equation(t, P, params):
    a,b,c,d,e = params[0],params[1],params[2],params[3],params[4]
    return a + b*P + c*P**2 + d/(P+e)
