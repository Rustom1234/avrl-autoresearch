import numpy as np
def equation(t, x, v, params):
    w=params[0]
    return params[1]+params[2]*np.cos(w*t)+params[3]*np.sin(w*t)+params[4]*x+params[5]*v+params[6]*x**2+params[7]*x**3+params[8]*v*np.abs(v)+params[9]*t
