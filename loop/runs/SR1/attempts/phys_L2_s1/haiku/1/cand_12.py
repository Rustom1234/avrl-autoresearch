import numpy as np
def equation(t, x, v, params):
    p=params
    return (p[0] + p[1]*x + p[2]*v + p[3]*x**3 + p[4]*x**2 + p[5]*v*np.abs(v)
            + p[6]*x*v + p[7]/v + p[8]*np.exp(-x) + p[9]*np.cos(x))
