import numpy as np
def equation(t, x, v, params):
    p=params
    return (p[0] + p[1]*x + p[2]*v + p[3]*x**3 + p[4]*np.exp(-p[5]*t)
            + p[6]*np.cos(p[7]*t) + p[8]*np.sin(p[7]*t) + p[9]*x*np.exp(-p[5]*t))
