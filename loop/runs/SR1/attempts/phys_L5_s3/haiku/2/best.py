import numpy as np
def equation(t, x, v, params):
    w=params[9]
    return params[0]+params[1]*x+params[2]*x**2+params[3]*x**3+params[4]*x**4+params[5]*x**5+params[6]*np.cos(w*t)+params[7]*np.sin(w*t)+params[8]*np.sin(3*w*t)