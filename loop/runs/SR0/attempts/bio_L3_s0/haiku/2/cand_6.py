import numpy as np
def equation(t, P, params):
    g = params[0]/P + params[1] + params[2]*P + params[3]*P**2
    return (params[4] + params[5]*t + params[6]*t**2)*g
