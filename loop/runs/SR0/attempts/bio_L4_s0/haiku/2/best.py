import numpy as np
def equation(t, P, params):
    return params[0]*P**params[1] + params[2]*P**params[3]*t + params[4]*t + params[5]*P*t**2 + params[6]*P**2*t
