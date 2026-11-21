
import numpy as np
def equation(t, x, v, params):
    p=params
    return (p[0] + p[1]*x + p[2]*x**2 + p[3]*t + p[4]*t**2 + p[5]/x + p[6]*t*x + p[7]*t*x**2 + p[8]*np.cos(x) + p[9]*v)
