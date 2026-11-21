import numpy as np
def equation(t, P, params):
    return params[0]/(1+params[1]*P) + params[2] + params[3]*P + params[4]*P**2