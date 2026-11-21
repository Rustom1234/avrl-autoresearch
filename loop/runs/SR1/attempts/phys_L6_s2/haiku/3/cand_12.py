import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]*t + params[2]*np.sqrt(t) + params[3]*x + params[4]*v
            + params[5]*x**2 + params[6]*v**2 + params[7]*x*v + params[8]*t*x + params[9]*v/x)
