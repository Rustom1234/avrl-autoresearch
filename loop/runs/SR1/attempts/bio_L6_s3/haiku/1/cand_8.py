import numpy as np
def equation(t,P,params):
    A,b,c,e,k,d=params[0],params[1],params[2],params[3],params[4],params[5]
    return -A*(1-np.exp(-b*t)) + P*(c+e*(1-np.exp(-k*t))) + d*P**2
