import numpy as np
def equation(t, P, params):
    return params[0]*P + params[1]*P**2 + params[2]*P**3 + params[3]*P**4 + params[4]/P + params[5]
