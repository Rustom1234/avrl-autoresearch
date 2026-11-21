import numpy as np
def equation(t, x, v, params):
    w = params[0]
    C = np.cos(w*t); S = np.sin(w*t)
    return (params[1] + params[2]*x + params[3]*v + params[4]*x**3 + params[5]*v**2
            + params[6]*C + params[7]*S + params[8]*v*C + params[9]*v*S)
