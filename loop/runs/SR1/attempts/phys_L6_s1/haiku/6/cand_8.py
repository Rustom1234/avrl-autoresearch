import numpy as np
def equation(t, x, v, params):
    w=params[0]
    return params[1] + params[2]*x + params[3]*v + (params[4]+params[5]*v)*np.sin(w*t) + (params[6]+params[7]*v)*np.cos(w*t) + params[8]*x*np.sin(w*t)+params[9]*x*np.cos(w*t)