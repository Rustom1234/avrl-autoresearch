import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*t + params[2]*t**2 + params[3]*x + params[4]*x**2 + params[5]*x**3 + params[6]*v + params[7]*np.log(x) + params[8]*x**2*np.log(x)
