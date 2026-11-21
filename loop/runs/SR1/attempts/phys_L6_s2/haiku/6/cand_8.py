import numpy as np
def equation(t, x, v, params):
    return params[0]*t**params[1] + params[2]*x + params[3]*v + params[4]*x**params[5] + params[6]*np.sign(v)*np.abs(v)**params[7] + params[8]*x*v + params[9]