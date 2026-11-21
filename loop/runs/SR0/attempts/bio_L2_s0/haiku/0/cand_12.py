import numpy as np
def equation(t,P,params):
    a,b,c,d,e,f=params[0],params[1],params[2],params[3],params[4],params[5]
    s=1-np.exp(-d*t)
    return a*P - b*P**2 - c*s - e*P*s - f*P*t
