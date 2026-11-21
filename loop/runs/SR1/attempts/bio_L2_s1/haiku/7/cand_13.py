import numpy as np
def equation(t, P, params):
    a,b,c,d,e,f,g,h,k = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]
    return a + b*P + c*P**2 + d*P**3 + e*t + f*t*P + g*t*P**2 + h*t**2 + k*np.sqrt(t)
