import numpy as np
def equation(t, P, params):
    a,b,c,d,e,f,g,h=params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return P*(a+b*np.exp(c*t)) + d*P**2 + e*P**3 + f + g*t + h*t*t