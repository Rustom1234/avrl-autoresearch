import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]*np.sqrt(t) + params[2]*x + params[3]*x**2 + params[4]*x**3
            + params[5]*np.sin(params[6]*x) + params[7]*np.cos(params[6]*x) + params[8]*v)
