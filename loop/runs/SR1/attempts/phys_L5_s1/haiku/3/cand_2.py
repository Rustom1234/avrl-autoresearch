import numpy as np
def equation(t, x, v, params):
    return params[0]*np.cos(params[1]*t)+params[2]*np.sin(params[1]*t)+params[3]*x+params[4]*v+params[5]*x**3+params[6]*v*np.abs(v)+params[7]*v**3+params[8]*x*v+params[9]