import numpy as np
def equation(t, x, v, params):
    w=params[9]
    return (params[0]*np.cos(w*t)+params[1]*np.sin(w*t)+params[2]*x+params[3]*x**3+params[4]*v
            +params[5]*x*v+params[6]*x**2+params[7]*x*np.cos(w*t)+params[8]*x*np.sin(w*t))
