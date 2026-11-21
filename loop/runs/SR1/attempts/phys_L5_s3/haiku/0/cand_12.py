import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]*np.cos(params[2]*t) + params[3]*np.sin(params[2]*t)
            + params[4]*x*np.exp(-params[5]*x**params[6]))
