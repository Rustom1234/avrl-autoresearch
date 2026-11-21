import numpy as np
def equation(t, x, v, params):
    w = params[0]
    s = params[1]
    for k in range(1,5):
        pass
    return (params[1] + params[2]*np.cos(w*t) + params[3]*np.sin(w*t)
            + params[4]*np.cos(2*w*t) + params[5]*np.sin(2*w*t)
            + params[6]*np.cos(3*w*t) + params[7]*np.sin(3*w*t)
            + params[8]*np.cos(4*w*t) + params[9]*np.sin(4*w*t))