import numpy as np
def equation(t, x, v, params):
    w=params[0]
    return params[1]*np.cos(w*t+params[2]) + params[3]*x + params[4]*x**3 + params[5]*v + params[6]*np.tanh(v) + params[7] + params[8]*np.sin(v) + params[9]*x*np.sin(w*t)
