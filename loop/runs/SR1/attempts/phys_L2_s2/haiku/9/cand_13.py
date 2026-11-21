import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]*x/v + params[2]*np.log(x)
            + params[3]*x**params[4] + params[5]*np.exp(-params[6]*x))
