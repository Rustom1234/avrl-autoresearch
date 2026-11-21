import numpy as np
def equation(t, P, params):
    return params[0]*P*(1 - P/(params[1] + params[2]*t)) + params[3]*P*t + params[4]*t + params[5]
