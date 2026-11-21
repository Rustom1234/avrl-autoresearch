import numpy as np
def equation(t, x, v, params):
    return (params[0]*np.cos(params[1]*t) + params[2]*np.sin(params[1]*t) + params[3]
            + params[4]*x + params[5]*x**2 + params[6]*x**3
            + params[7]*v + params[8]*v**2 + params[9]/v)
