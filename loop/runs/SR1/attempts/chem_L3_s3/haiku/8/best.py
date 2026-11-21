import numpy as np
def equation(t, A, params):
    return (params[0]*t**(-params[1]) + params[2] + params[3]*t + params[4]*t**2
            + A*(params[5] + params[6]*t + params[7]*t**2) + params[8]*A**2)
