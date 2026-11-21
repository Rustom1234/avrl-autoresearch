import numpy as np
def equation(t, P, params):
    a,b,K,c,d,e = params[0],params[1],params[2],params[3],params[4],params[5]
    return P*(a + b*t)*(1 - P/(K + c*t)) - d/P - e*t
