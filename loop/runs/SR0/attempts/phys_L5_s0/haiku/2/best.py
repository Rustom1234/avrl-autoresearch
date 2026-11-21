import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]/t + params[2]/v + params[3]*t + params[4]*t**2
            + params[5]*v + params[6]*v*v + params[7]*v**3 + params[8]*x + params[9]*x**3)