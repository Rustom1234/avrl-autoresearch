import numpy as np
def equation(t,P,params):
    r,A,K,H,b=params[0],params[1],params[2],params[3],params[4]
    return r*P*(P/A-1)*(1-P/K) - H*(1-np.exp(-b*t))
