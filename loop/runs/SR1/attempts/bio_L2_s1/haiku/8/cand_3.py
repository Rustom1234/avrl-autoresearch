import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*t + params[2]*P + params[3]*P**2 + params[4]*t*P + params[5]*t**2 + params[6]*t*P**2 + params[7]*t**2*P
