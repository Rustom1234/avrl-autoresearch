import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]/x + params[2]*np.sqrt(t) + params[3]*t
            + params[4]*t**1.5 + params[5]*t**2)