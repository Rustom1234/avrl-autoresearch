import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*t + params[2]/P + params[3]*P + params[4]*P**2 + params[5]*P**3 + params[6]*t*P + params[7]*t*P**2 + params[8]*t/P + params[9]*t**2
