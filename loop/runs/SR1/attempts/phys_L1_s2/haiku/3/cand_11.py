import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]*np.sqrt(t) + params[2]*x + params[3]*x**2
            + params[4]*np.exp(-params[5]*x))
