import numpy as np
def equation(t, A, params):
    c0,c1,p1,c2,p2,c3,s,m = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return c0 + c1*A**p1 + c2*A**p2 + c3*t**s*A**m