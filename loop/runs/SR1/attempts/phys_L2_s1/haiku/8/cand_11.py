import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]*x**params[2] + params[3]*v + params[4]*v**2 + params[5]*x*v**2
            + params[6]/v + params[7]*np.cos(x) + params[8]*np.sin(x*v))