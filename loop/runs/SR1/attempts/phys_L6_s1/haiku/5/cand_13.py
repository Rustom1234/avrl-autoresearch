import numpy as np
def equation(t, x, v, params):
    a, w = params[0], params[1]
    e = np.exp(a*t)
    return (params[2] + params[3]*x + params[4]*v + params[5]*v*np.abs(v)
            + e*(params[6]*np.cos(w*t) + params[7]*np.sin(w*t))
            + params[8]*t + params[9]*t*v)
