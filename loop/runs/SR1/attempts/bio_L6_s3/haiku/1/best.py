import numpy as np
def equation(t,P,params):
    A,b,c,d,e,f,g,h=params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return -A*(1-np.exp(-b*t)) + P*(c+d*t) + P**2*(e+f*t) + g*P**3 + h*P**3*t
