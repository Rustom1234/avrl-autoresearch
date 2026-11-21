import numpy as np
def equation(t, P, params):
    return params[0]*P + params[1]*P**1.5 + params[2]*P**2 + params[3]*P**2.5 + params[4]*P**3
