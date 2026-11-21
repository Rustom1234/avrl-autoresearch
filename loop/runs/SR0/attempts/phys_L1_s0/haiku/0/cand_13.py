
import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]/v + params[2]*t + params[3]*t**2 + params[4]*t**3 + params[5]*np.sin(t) + params[6]*np.cos(t) + params[7]*x
