import numpy as np
def equation(t, x, v, params):
    w=params[9]
    return (params[0]*np.cos(w*t)+params[1]*np.sin(w*t)+params[2]*v+params[3]*x
            +params[4]*x**3+params[5]*v*np.abs(v)+params[6]
            +params[7]*v*np.cos(w*t)+params[8]*v*np.sin(w*t))
