import numpy as np
def equation(t, P, params):
    a,b,c,d,e,f,g = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return a*P + b*t*P + c*P**2 + d*t*P**2 + e*t + f*t**2 + g
