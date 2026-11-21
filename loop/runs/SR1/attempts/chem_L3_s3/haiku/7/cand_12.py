import numpy as np
def equation(t, A, params):
    a,b,c,d,e,f = params[0],params[1],params[2],params[3],params[4],params[5]
    return a*t**b + c + d*A**2*t**e + f*A
