import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]*x + params[2]*x**3 + params[3]*x*v*v + params[4]/v + params[5]*x*x*v + params[6]/x + params[7]*x*x + params[8]*np.cos(params[9]*t))