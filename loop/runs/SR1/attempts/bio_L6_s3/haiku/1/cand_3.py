import numpy as np
def equation(t,P,params):
    a,K,b,c,d=params[0],params[1],params[2],params[3],params[4]
    return a*P*(1-P/K) - b*P*(1-np.exp(-c*t)) - d*P**2
