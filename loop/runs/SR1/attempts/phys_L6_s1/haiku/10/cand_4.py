import numpy as np
def equation(t, x, v, params):
    w = params[0]
    return (params[1]*np.sin(w*t) + params[2]*np.cos(w*t) + params[3]
            - params[4]*np.abs(x)**params[5] - params[6]*np.abs(v)**params[7] + params[8]*x*v + params[9]*x)
