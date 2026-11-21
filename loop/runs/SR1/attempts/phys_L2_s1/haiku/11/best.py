import numpy as np
def equation(t, x, v, params):
    p=params
    return (p[0] + p[1]*x + p[2]*x**3 + p[3]*v + p[4]/v + p[5]*x*v + p[6]/x
            + p[7]*x/v + p[8]*v**3 + p[9]*np.cos(9.9*t))
