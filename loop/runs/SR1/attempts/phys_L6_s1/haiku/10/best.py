import numpy as np
def equation(t, x, v, params):
    w = params[0]
    s = np.sin(w*t); c = np.cos(w*t); s2 = np.sin(2*w*t); c2 = np.cos(2*w*t)
    return (params[1] + params[2]*v + params[3]*s + params[4]*c + params[5]*v*s
            + params[6]*v*c + params[7]*s2 + params[8]*c2 + params[9]*v*s2)
