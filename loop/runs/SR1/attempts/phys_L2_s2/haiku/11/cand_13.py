import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]*x/v + params[2]*np.log(x) + params[3]/x
            + params[4]*x + params[5]*x**2 + params[6]/x**2 + params[7]*np.log(x)/x
            + params[8]*x**3 + params[9]*np.log(x)*x)