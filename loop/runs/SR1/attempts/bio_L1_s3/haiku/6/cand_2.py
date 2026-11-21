import numpy as np
def equation(t,P,params):
    a,b,c,d=params[0],params[1],params[2],params[3]
    return a*(1-np.exp(-b*P))*np.exp(-c*P)+d
