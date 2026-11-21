import numpy as np
def equation(t, x, v, params):
    a, w = params[0], params[1]
    e = np.exp(a*t)
    return (params[2] + params[3]*x + params[4]*v + params[5]*x**3 + params[6]*v*np.abs(v)
            + e*(params[7]*np.cos(w*t) + params[8]*np.sin(w*t) + v*(params[9]*np.cos(w*t)))
            + params[10]*t + params[11]*t*v)
