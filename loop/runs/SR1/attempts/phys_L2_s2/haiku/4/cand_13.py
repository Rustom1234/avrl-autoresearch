import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]*x**3 + params[3]*v + params[4]*x/v + params[5]*np.log(x) + params[6]*np.exp(-x) + params[7]*x*np.log(x) + params[8]*np.cos(t) + params[9]*v*np.exp(-x)