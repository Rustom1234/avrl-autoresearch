import numpy as np
def equation(t, P, params):
    return (params[0] + params[1]*t + params[2]*t**2 + params[3]*t**3
            + P*(params[4] + params[5]*t + params[6]*t**2)
            + params[7]*P**2 + params[8]*P**2*t + params[9]*P**3)
