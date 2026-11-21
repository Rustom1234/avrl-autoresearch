import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]*x + params[2]*v + params[3]*x*v**2 + params[4]/v
            + params[5]*np.sqrt(x) + params[6]*np.cos(x) + params[7]*x*v + params[8]*np.log(v) + params[9]*v**2)