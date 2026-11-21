import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]/t + params[2]*np.sqrt(x) + params[3]*x/v + params[4]*x + params[5]*v + params[6]*x**2 + params[7]*v*np.abs(v)
