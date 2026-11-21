import numpy as np
def equation(t, A, params):
    a,b,c,d,e,f,g,h = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return a*t**b + c*A*t**d + e*A**2*t**f + g + h*A
