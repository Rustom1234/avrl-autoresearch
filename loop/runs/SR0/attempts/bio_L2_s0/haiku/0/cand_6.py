import numpy as np
def equation(t,P,params):
    a,b,c,d,e=params[0],params[1],params[2],params[3],params[4]
    return a*P - b*P**2 - c*t**d + e*P*t
