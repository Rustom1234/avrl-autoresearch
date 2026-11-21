import numpy as np
def equation(t, P, params):
    a0,a1,b0,b1,c0,c1,c2,d0 = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    r = a0 + a1*t
    K = b0 + b1*t
    return r*P*(1-P/K) + c0 + c1*t + c2*t*t + d0*P*P*P
