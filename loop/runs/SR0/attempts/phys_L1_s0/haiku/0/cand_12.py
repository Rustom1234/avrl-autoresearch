
import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]/v + params[2]*t + params[3]*np.sin(params[4]*t) + params[5]*np.cos(params[4]*t) + params[6]*x
