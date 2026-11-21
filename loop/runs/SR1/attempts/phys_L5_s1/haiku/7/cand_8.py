import numpy as np
def equation(t, x, v, params):
    w=params[0]
    return params[1]+params[2]*x+params[3]*x**3+params[4]*np.abs(v)**params[5]+params[6]*np.cos(w*t)+params[7]*np.sin(w*t)+params[8]*t+params[9]*np.sign(v)*np.abs(v)