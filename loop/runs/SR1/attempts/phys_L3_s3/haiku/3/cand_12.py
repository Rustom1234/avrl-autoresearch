import numpy as np
def equation(t, x, v, params):
    return (params[0] + params[1]*x + params[2]*v + params[3]*x**2 + params[4]*x*v + params[5]*v**2
            + params[6]*v**3 + params[7]*x*v**2 + params[8]*t + params[9]*t**2)
