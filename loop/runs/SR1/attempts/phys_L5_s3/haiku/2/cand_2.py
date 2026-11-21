import numpy as np
def equation(t, x, v, params):
    w=params[6]
    return params[0]+params[1]*x+params[2]*x**2+params[3]*x**3+params[4]*v+params[5]*v*np.abs(v)+params[7]*np.cos(w*t)+params[8]*np.sin(w*t)