import numpy as np
def equation(t, P, params):
    return P*(params[0] + params[1]*t**2) + P**2*(params[2] + params[3]*t + params[4]*t**2) + params[5] + params[6]*t**2 + params[7]*t
