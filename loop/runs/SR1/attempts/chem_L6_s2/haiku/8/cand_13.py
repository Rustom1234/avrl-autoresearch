import numpy as np
def equation(t, A, params):
    return (params[0] + params[1]/A + params[2]*A + params[3]*A**2
            + params[4]*t**2/A + params[5]*t + params[6]*t*A + params[7]*t**2 + params[8]*t**2*A**2/1.0 + params[9]*t/A)
