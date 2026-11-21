import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]/x + params[2]*v + params[3]*v*np.abs(v)
            + params[4]*t + params[5]*np.exp(-params[6]*t) + params[7]/t
            + params[8]/t**2 + params[9]*t**2)