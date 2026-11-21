import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]*v + params[2]*t + params[3]*t*np.tanh(x) + params[4]*np.cos(x)
            + params[5]*np.tanh(x) + params[6]*v**3 + params[7]*t*v + params[8]*np.sin(t) + params[9]*x*v)