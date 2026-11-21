import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]/t + params[2]*x + params[3]*x**3 + params[4]*v + params[5]*v*np.abs(v) + params[6]*x*v + params[7]*np.sqrt(x)
