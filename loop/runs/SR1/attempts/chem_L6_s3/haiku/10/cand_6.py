import numpy as np
def equation(t, A, params):
    p0,p1,p2,p3,p4 = params[0],params[1],params[2],params[3],params[4]
    return p0*np.exp(-p1*t) - p2*t**p3*np.abs(A)**p4
