import numpy as np
def equation(t, P, params):
    a,b,c,d,e,f=params[0],params[1],params[2],params[3],params[4],params[5]
    return (a+b*t)*np.exp(-c*P) + d*t*P*np.exp(-e*P) + f*P
