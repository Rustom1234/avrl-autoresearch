import numpy as np
def equation(t, P, params):
    return (params[0] + params[1]*P + params[2]*P**2 + params[3]*P**3
            + t*(params[4] + params[5]*P + params[6]*P**2)
            + t**2*(params[7] + params[8]*P + params[9]*P**2))