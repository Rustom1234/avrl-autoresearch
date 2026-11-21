import numpy as np
def equation(t,P,params):
    a,r,K=params[0],params[1],params[2]
    return a + r*P*(1-P/K)
