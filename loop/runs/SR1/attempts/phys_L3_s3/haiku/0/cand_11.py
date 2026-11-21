import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]*x + params[2]*x*v + params[3]*t + params[4]*np.exp(-t)
            + params[5]*np.log(v) + params[6]*v + params[7]*x**2 + params[8]*np.cos(params[9]*t))