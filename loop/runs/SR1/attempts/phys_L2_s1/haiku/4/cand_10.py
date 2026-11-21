import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]*v + params[3]*x**3 + params[4]*np.abs(v)**params[5]*np.sign(v) + params[6]*x**params[7] + params[8]*v**params[9]