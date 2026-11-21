import numpy as np
def equation(t, P, params):
    return params[0] + params[1]*P**2 + params[2]*P/(params[3]+P)
