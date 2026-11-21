import numpy as np
def equation(t, x, v, params):
    p=params
    return (p[0]+p[1]*t)*x + p[2]*x**2 + p[3]*x**3 + (p[4]+p[5]*t)*v + p[6] + p[7]*t + p[8]*t*v*v + p[9]*t*x*x + 0*t*t
