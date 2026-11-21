import numpy as np
def equation(t, P, params):
    a,b,c,d,e,f,g,h=params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return (a+b*t)*np.exp(-c*P) + (d+e*t)*P + f*P**2 + g*t + h
