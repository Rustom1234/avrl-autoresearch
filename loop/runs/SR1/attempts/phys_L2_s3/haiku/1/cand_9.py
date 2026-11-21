import numpy as np
def equation(t, x, v, params):
    p=params
    return (p[0] + p[1]*(1-np.exp(-p[2]*t)) + p[3]*x + p[4]*x**2 + p[5]*x**3 + p[6]*v + p[7]*v**2 + p[8]*x*v)
