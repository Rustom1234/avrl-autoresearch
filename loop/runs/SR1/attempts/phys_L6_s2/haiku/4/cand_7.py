import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*v + params[2]*v**3 + params[3]*t + params[4]*t*x + params[5]*np.sin(x) + params[6]*np.sin(v) + params[7]*x*v + params[8]*x**2 + params[9]*np.cos(t)