import numpy as np
def equation(t,P,params):
    a,b,c,k1,k2,d,e=params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return (a+b*np.exp(-k1*t))*P + c*P**2 + d*(1-np.exp(-k2*t))
