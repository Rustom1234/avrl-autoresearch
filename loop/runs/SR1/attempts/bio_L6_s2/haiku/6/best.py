import numpy as np
def equation(t, P, params):
    return P*(params[0] + params[1]*P + params[2]*t + params[3]*t**2 + params[4]*P*t + params[5]*P*t**2 + params[6]*P**2)