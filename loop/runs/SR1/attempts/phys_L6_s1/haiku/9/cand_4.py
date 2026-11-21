import numpy as np
def equation(t, x, v, params):
    w=params[5]
    return params[0]+params[1]*x+params[2]*v+(params[3]+params[4]*v)*np.cos(w*t)+(params[6]+params[7]*v)*np.sin(w*t)+params[8]*x*np.cos(w*t)+params[9]*x*np.sin(w*t)
