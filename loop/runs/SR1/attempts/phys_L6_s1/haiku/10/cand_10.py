import numpy as np
def equation(t, x, v, params):
    w = params[0]
    s = np.sin(w*t); c = np.cos(w*t)
    e = np.exp(-params[9]*t)
    return (params[1] + params[2]*v + e*(params[3]*s + params[4]*c
            + params[5]*v*s + params[6]*v*c) + params[7]*x + params[8]*x*v)
