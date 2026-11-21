import numpy as np
def equation(t, x, v, params):
    w = params[9]
    c = np.cos(w*t); s = np.sin(w*t)
    return (params[0] + params[1]*x + params[2]*v + params[3]*x**3
            + params[4]*v*np.abs(v)
            + (params[5] + params[6]*x)*c + (params[7] + params[8]*v)*s)