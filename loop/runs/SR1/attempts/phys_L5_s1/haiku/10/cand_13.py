import numpy as np
def equation(t, x, v, params):
    p0,p1,p2,p3,p4,p5,p6,p7,p8,p9 = params
    return p0 + p1*np.exp(-p2*t) + p3*t + p4*t**2 + p5*v + p6*v*np.abs(v) + p7*x + p8*x**2 + p9*x**3
