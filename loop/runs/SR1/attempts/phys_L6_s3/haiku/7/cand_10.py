import numpy as np
def equation(t, x, v, params):
    w=params[0]
    return (params[1]*np.sin(w*t)+params[2]*np.cos(w*t) + params[3]
            - params[4]*np.tanh(params[5]*v) - params[6]*x - params[7]*x**3 - params[8]*v - params[9]*x*v)
