import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]*t + params[2]*v/x + params[3]*np.exp(-v)
            + params[4]*t*x + params[5]*x*v + params[6]*x**2 + params[7]/t)
