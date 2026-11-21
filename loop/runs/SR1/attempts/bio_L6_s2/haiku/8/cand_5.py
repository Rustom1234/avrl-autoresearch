import numpy as np
def equation(t, P, params):
    return params[0]*P + params[1]*P**params[2] + params[3]*t*P + params[4]*t*P**2 + params[5]*t
