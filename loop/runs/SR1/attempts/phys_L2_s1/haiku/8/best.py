import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]*np.log(v) + params[2]*x*v**2 + params[3]/v + params[4]*np.cos(x)
            + params[5]*np.sqrt(x) + params[6]*x*v + params[7]*np.sin(x*v) + params[8]/x)