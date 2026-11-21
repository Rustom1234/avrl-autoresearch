import numpy as np
def equation(t, x, v, params):
    w=params[1]
    s=np.sin(w*t); c=np.cos(w*t)
    return params[0]*s+params[2]+params[3]*t+params[4]*v+params[5]*v*np.abs(v)+params[6]*v**3+params[7]*v*s+params[8]*v*c+params[9]*c
