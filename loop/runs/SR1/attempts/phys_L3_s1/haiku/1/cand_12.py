import numpy as np
def equation(t, x, v, params):
    w = params[9]
    c = np.cos(w*t); s = np.sin(w*t)
    return (params[0] + params[1]*x + params[2]*v + params[3]*x**3
            + params[4]*v*np.abs(v) + params[5]*x*v
            + params[6]*c + params[7]*s + params[8]*c*x + params[10]*s*x + params[11]*c*v + params[12]*s*v)