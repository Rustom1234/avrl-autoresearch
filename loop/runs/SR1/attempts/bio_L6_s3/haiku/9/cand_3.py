import numpy as np
def equation(t, P, params):
    a,b,c,d,k,e=params[0],params[1],params[2],params[3],params[4],params[5]
    g=np.exp(-k*t)
    return a*g + b*P*g + c*P**2*g + d*(1-g) + e*P*(1-g)
