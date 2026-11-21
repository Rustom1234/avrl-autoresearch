import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]*x + params[2]*v + params[3]*x*v + params[4]*x**2
            + params[5]*np.log(v) + params[6]*x**3 + params[7]*v**2 + params[8]*t + params[9]*np.exp(-t))